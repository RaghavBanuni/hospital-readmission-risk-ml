"""Command line interface: ``python -m readmission.cli <command>``."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import CohortConfig, generate_encounters, readmission_rate_by
from .evaluate import (
    OutreachEconomics,
    best_net_benefit,
    calibration_table,
    capacity_metrics,
    discrimination_metrics,
    expected_calibration_error,
    net_benefit_table,
    threshold_for_capacity,
)
from .fairness import (
    AUDIT_ATTRIBUTES_DEFAULT,
    fairness_gaps,
    subgroup_metrics,
    threshold_per_group,
    worst_equal_opportunity_gap,
)
from .features import AUDIT_ATTRIBUTES, FEATURE_CONTRACT, TARGET, with_audit_columns
from .model import (
    coefficient_table,
    fit_calibrated,
    fit_model,
    permutation_importance_table,
    predict_risk,
    three_way_split,
)


@dataclass
class Selection:
    """Outcome of model selection on the validation split."""

    name: str
    model: Any
    threshold: float
    comparison: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    test_risk: np.ndarray


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path}")


def _cohort(args: argparse.Namespace) -> pd.DataFrame:
    frame = generate_encounters(
        CohortConfig(n_patients=args.patients, target_rate=args.rate, seed=args.seed)
    )
    return with_audit_columns(frame)


def _select_model(args: argparse.Namespace, cohort: pd.DataFrame) -> Selection:
    """Fit candidates on train, choose on validation, score test once."""
    train, validation, test = three_way_split(cohort, seed=args.seed)
    candidates: dict[str, Any] = {
        "logistic": fit_model(train, "logistic", seed=args.seed),
        "gradient_boosting": fit_model(train, "gradient_boosting", seed=args.seed),
    }
    if not args.no_calibration:
        candidates["gradient_boosting_calibrated"] = fit_calibrated(
            train, "gradient_boosting", seed=args.seed
        )

    validation_target = validation[TARGET].to_numpy(dtype=int)
    rows = []
    for name, model in candidates.items():
        risk = predict_risk(model, validation)
        rows.append(
            {
                "model": name,
                **discrimination_metrics(validation_target, risk),
                "expected_calibration_error": expected_calibration_error(validation_target, risk),
                **{
                    f"capacity_{key}": value
                    for key, value in capacity_metrics(
                        validation_target, risk, args.capacity
                    ).items()
                    if key in {"precision", "recall", "lift_over_random"}
                },
            }
        )
    comparison = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    champion_name = str(comparison.iloc[0]["model"])
    champion = candidates[champion_name]

    validation_risk = predict_risk(champion, validation)
    threshold = threshold_for_capacity(validation_risk, args.capacity)

    return Selection(
        name=champion_name,
        model=champion,
        threshold=threshold,
        comparison=comparison.reset_index(drop=True),
        train=train,
        validation=validation,
        test=test,
        test_risk=predict_risk(champion, test),
    )


def cmd_data(args: argparse.Namespace) -> int:
    """Generate a cohort and describe its readmission structure."""
    cohort = _cohort(args)
    print(
        f"{len(cohort)} discharges, readmission rate "
        f"{cohort[TARGET].mean():.4f}\n"
    )
    for column in ("admission_type", "discharge_disposition", "insurance", "age_band"):
        print(f"readmission rate by {column}")
        print(readmission_rate_by(cohort, column).to_string(index=False), "\n")
    print("follow-up appointment booked, by payer")
    print(
        cohort.groupby("insurance")["follow_up_booked"].mean().round(4).to_string(),
        "\n",
    )
    missing = cohort.isna().sum()
    print("missing values per column")
    print(missing[missing > 0].to_string(), "\n")

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(destination, index=False)
    print(f"wrote {destination}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train, calibrate, choose an operating point and report on the test split."""
    out_dir = Path(args.out)
    cohort = _cohort(args)
    selection = _select_model(args, cohort)
    economics = OutreachEconomics(
        outreach_cost=args.outreach_cost,
        readmission_cost=args.readmission_cost,
        prevention_rate=args.prevention_rate,
    )

    print(
        f"split: train={len(selection.train)} validation={len(selection.validation)} "
        f"test={len(selection.test)}\n"
    )
    print("candidate models on the validation split")
    print(selection.comparison.to_string(index=False), "\n")
    print(f"champion: {selection.name}")
    print(
        f"threshold at {args.capacity:.1%} outreach capacity: {selection.threshold:.4f}\n"
    )

    test_target = selection.test[TARGET].to_numpy(dtype=int)
    metrics = discrimination_metrics(test_target, selection.test_risk)
    metrics["expected_calibration_error"] = expected_calibration_error(
        test_target, selection.test_risk
    )
    print("held-out test metrics")
    for key, value in metrics.items():
        print(f"  {key:<28}{value}")

    reach = capacity_metrics(test_target, selection.test_risk, args.capacity)
    print("\nwhat the outreach list reaches (test split)")
    for key, value in reach.items():
        print(f"  {key:<28}{value}")

    benefit = net_benefit_table(test_target, selection.test_risk, economics)
    chosen = best_net_benefit(benefit)
    print("\nhighest net benefit operating point")
    for key in ("threshold", "contact_rate", "precision", "recall", "net_benefit"):
        print(f"  {key:<28}{chosen[key]}")
    print()

    calibration = calibration_table(test_target, selection.test_risk)
    print("calibration on the test split")
    print(calibration.to_string(index=False), "\n")

    importance = permutation_importance_table(selection.model, selection.test, seed=args.seed)
    print("top drivers by permutation importance")
    print(importance.head(12).to_string(index=False), "\n")

    _write(selection.comparison, out_dir / "model_comparison.csv")
    _write(calibration, out_dir / "calibration.csv")
    _write(pd.DataFrame([reach]), out_dir / "capacity_metrics.csv")
    _write(benefit, out_dir / "net_benefit.csv")
    _write(importance, out_dir / "feature_importance.csv")

    logistic = fit_model(selection.train, "logistic", seed=args.seed)
    _write(coefficient_table(logistic), out_dir / "logistic_coefficients.csv")

    report = {
        "champion": selection.name,
        "capacity_fraction": float(args.capacity),
        "threshold": round(float(selection.threshold), 6),
        "feature_contract": FEATURE_CONTRACT.as_dict(),
        "economics": economics.as_dict(),
        "test_metrics": metrics,
        "capacity_metrics": reach,
        "best_net_benefit": chosen,
    }
    report_path = out_dir / "training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Fairness audit of the champion at the deployed threshold."""
    out_dir = Path(args.out)
    cohort = _cohort(args)
    selection = _select_model(args, cohort)
    test_target = selection.test[TARGET].to_numpy(dtype=int)

    print(f"champion: {selection.name}, threshold {selection.threshold:.4f}")
    print(f"audited attributes: {', '.join(AUDIT_ATTRIBUTES)}\n")

    metrics = subgroup_metrics(
        selection.test,
        test_target,
        selection.test_risk,
        selection.threshold,
        attributes=AUDIT_ATTRIBUTES,
        min_group_size=args.min_group_size,
    )
    columns = [
        "attribute",
        "group",
        "n",
        "prevalence",
        "selection_rate",
        "true_positive_rate",
        "false_negative_rate",
        "precision",
        "roc_auc",
        "calibration_in_the_large",
    ]
    print("performance by subgroup")
    print(metrics[columns].to_string(index=False), "\n")

    gaps = fairness_gaps(metrics)
    print("gaps between best and worst group (groups below the size floor excluded)")
    print(gaps.to_string(index=False), "\n")

    worst = worst_equal_opportunity_gap(gaps)
    print("largest equal-opportunity gap")
    for key, value in worst.items():
        print(f"  {key:<28}{value}")
    print()

    parity = threshold_per_group(
        selection.test, selection.test_risk, args.parity_attribute, args.capacity
    )
    print(
        f"thresholds that would equalise the selection rate across "
        f"'{args.parity_attribute}' (a remedy to consider, not a default)"
    )
    print(parity.to_string(index=False), "\n")

    _write(metrics, out_dir / "subgroup_metrics.csv")
    _write(gaps, out_dir / "fairness_gaps.csv")
    _write(parity, out_dir / "group_thresholds.csv")

    audit_path = out_dir / "fairness_report.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "champion": selection.name,
                "threshold": round(float(selection.threshold), 6),
                "capacity_fraction": float(args.capacity),
                "min_group_size": int(args.min_group_size),
                "largest_equal_opportunity_gap": worst,
                "gaps": gaps.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {audit_path}")
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--patients", type=int, default=30_000)
    parser.add_argument("--rate", type=float, default=0.15, help="target readmission rate")
    parser.add_argument("--seed", type=int, default=47)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_arguments(parser)
    parser.add_argument(
        "--capacity",
        type=float,
        default=0.08,
        help="share of discharges the transition team can contact",
    )
    parser.add_argument("--no-calibration", dest="no_calibration", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readmission",
        description="30-day readmission risk with capacity-aware evaluation and a fairness audit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="generate and describe a cohort")
    _add_common_arguments(data_parser)
    data_parser.add_argument("--out", default="data/encounters.csv")
    data_parser.set_defaults(func=cmd_data)

    train_parser = subparsers.add_parser("train", help="train, calibrate and evaluate")
    _add_model_arguments(train_parser)
    train_parser.add_argument("--outreach-cost", dest="outreach_cost", type=float, default=120.0)
    train_parser.add_argument(
        "--readmission-cost", dest="readmission_cost", type=float, default=11_000.0
    )
    train_parser.add_argument(
        "--prevention-rate", dest="prevention_rate", type=float, default=0.20
    )
    train_parser.add_argument("--out", default="reports")
    train_parser.set_defaults(func=cmd_train)

    audit_parser = subparsers.add_parser("audit", help="subgroup fairness audit")
    _add_model_arguments(audit_parser)
    audit_parser.add_argument(
        "--min-group-size", dest="min_group_size", type=int, default=100
    )
    audit_parser.add_argument(
        "--parity-attribute",
        dest="parity_attribute",
        default="insurance",
        choices=list(AUDIT_ATTRIBUTES_DEFAULT),
    )
    audit_parser.add_argument("--out", default="reports")
    audit_parser.set_defaults(func=cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
