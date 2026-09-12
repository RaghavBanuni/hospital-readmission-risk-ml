"""Subgroup fairness audit.

What this module reports, and why each one is here:

* **selection rate** per group at the deployed threshold - who actually gets the
  intervention (demographic parity view);
* **false-negative rate** per group - whose high-risk patients the model misses
  (equal-opportunity view, the one that maps to clinical harm);
* **per-group calibration** - a group whose risk is systematically
  under-predicted will be under-served no matter where the threshold sits;
* **discrimination within group** (AUC, average precision) - a model can rank well
  overall while ranking almost randomly inside a smaller group.

Two honesty rules are enforced in code:

1. groups smaller than ``min_group_size`` are reported but excluded from gap
   calculations, because a 12-patient group produces spectacular fake disparities;
2. metrics that are undefined for a group (no positives, no negatives) come back
   as ``NaN`` instead of a silent zero.

Demographic parity and equal opportunity cannot both be satisfied when base rates
differ between groups.  The audit therefore reports both gaps and leaves the
trade-off visible rather than collapsing it into one "fairness score".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .evaluate import calibration_in_the_large, expected_calibration_error
from .features import AUDIT_ATTRIBUTES

#: Attributes audited by default - re-exported so callers (and the CLI) have one
#: source of truth for the audit dimensions.
AUDIT_ATTRIBUTES_DEFAULT: tuple[str, ...] = AUDIT_ATTRIBUTES

MIN_GROUP_SIZE: int = 100

GAP_METRICS: tuple[str, ...] = (
    "selection_rate",
    "false_negative_rate",
    "true_positive_rate",
    "precision",
    "roc_auc",
    "calibration_in_the_large",
)


def _group_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Metrics for one subgroup; undefined quantities are NaN, never zero."""
    flagged = probabilities >= threshold
    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())
    true_positives = int(((y_true == 1) & flagged).sum())
    false_positives = int(((y_true == 0) & flagged).sum())
    contacted = int(flagged.sum())

    both_classes = positives > 0 and negatives > 0
    return {
        "n": int(y_true.size),
        "prevalence": round(float(y_true.mean()), 5),
        "mean_predicted_risk": round(float(probabilities.mean()), 5),
        "selection_rate": round(contacted / y_true.size, 5),
        "true_positive_rate": round(true_positives / positives, 5) if positives else float("nan"),
        "false_negative_rate": round(1.0 - true_positives / positives, 5)
        if positives
        else float("nan"),
        "false_positive_rate": round(false_positives / negatives, 5)
        if negatives
        else float("nan"),
        "precision": round(true_positives / contacted, 5) if contacted else float("nan"),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 5)
        if both_classes
        else float("nan"),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 5)
        if both_classes
        else float("nan"),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 5),
        "calibration_in_the_large": calibration_in_the_large(y_true, probabilities)
        if both_classes and y_true.size >= 20
        else float("nan"),
        "expected_calibration_error": expected_calibration_error(y_true, probabilities, n_bins=5)
        if both_classes and y_true.size >= 20
        else float("nan"),
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true,
    probabilities,
    threshold: float,
    attributes: tuple[str, ...] = AUDIT_ATTRIBUTES_DEFAULT,
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """One row per (attribute, group) with performance inside that group."""
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(probabilities, dtype=float).ravel()
    if not len(frame) == y.size == p.size:
        raise ValueError("frame, labels and probabilities must align")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    missing = [attribute for attribute in attributes if attribute not in frame.columns]
    if missing:
        raise ValueError(f"audit attributes not present in the frame: {missing}")

    rows: list[dict[str, object]] = []
    for attribute in attributes:
        values = frame[attribute].astype(str).to_numpy()
        for group in sorted(pd.unique(values)):
            mask = values == group
            metrics = _group_metrics(y[mask], p[mask], threshold)
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "included_in_gaps": bool(metrics["n"] >= min_group_size),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def fairness_gaps(
    metrics: pd.DataFrame, gap_metrics: tuple[str, ...] = GAP_METRICS
) -> pd.DataFrame:
    """Worst-group minus best-group spread per attribute and metric.

    ``selection_rate`` spread is the demographic-parity gap;
    ``false_negative_rate`` spread is the equal-opportunity gap - the one to read
    first, because it counts patients whose risk the model failed to flag.
    """
    if metrics.empty:
        raise ValueError("subgroup metrics table is empty")
    eligible = metrics[metrics["included_in_gaps"]]
    if eligible.empty:
        raise ValueError(
            "no subgroup is large enough for a gap calculation; lower min_group_size "
            "or evaluate on more encounters"
        )

    rows: list[dict[str, object]] = []
    for attribute, group_frame in eligible.groupby("attribute", sort=True):
        for metric in gap_metrics:
            if metric not in group_frame.columns:
                continue
            series = group_frame[["group", metric]].dropna()
            if len(series) < 2:
                continue
            highest = series.loc[series[metric].idxmax()]
            lowest = series.loc[series[metric].idxmin()]
            # for an error rate the *highest* value is the worst outcome; for a
            # performance metric it is the lowest
            higher_is_worse = metric in {"false_negative_rate", "false_positive_rate"}
            rows.append(
                {
                    "attribute": attribute,
                    "metric": metric,
                    "groups_compared": int(len(series)),
                    "best_group": str(lowest["group"] if higher_is_worse else highest["group"]),
                    "worst_group": str(highest["group"] if higher_is_worse else lowest["group"]),
                    "max_value": round(float(highest[metric]), 5),
                    "min_value": round(float(lowest[metric]), 5),
                    "gap": round(float(highest[metric] - lowest[metric]), 5),
                    "ratio": round(float(highest[metric] / lowest[metric]), 3)
                    if float(lowest[metric]) != 0.0
                    else float("inf"),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["metric", "gap"], ascending=[True, False])
        .reset_index(drop=True)
    )


def worst_equal_opportunity_gap(gaps: pd.DataFrame) -> dict[str, object]:
    """The largest false-negative-rate spread across all audited attributes."""
    subset = gaps[gaps["metric"] == "false_negative_rate"]
    if subset.empty:
        raise ValueError("no false-negative-rate gap could be computed")
    worst = subset.loc[subset["gap"].idxmax()]
    return {
        "attribute": str(worst["attribute"]),
        "worst_group": str(worst["worst_group"]),
        "best_group": str(worst["best_group"]),
        "false_negative_rate_gap": float(worst["gap"]),
    }


def threshold_per_group(
    frame: pd.DataFrame,
    probabilities,
    attribute: str,
    capacity_fraction: float = 0.08,
) -> pd.DataFrame:
    """Group-specific thresholds that equalise the selection rate.

    Offered as a remedy to *inspect*, not a default: equalising selection rates
    across groups with different prevalence deliberately trades some overall
    efficiency for parity, and that is a clinical and ethical decision rather than
    an engineering one.
    """
    if attribute not in frame.columns:
        raise ValueError(f"unknown attribute: {attribute}")
    if not 0.0 < capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be in (0, 1]")
    p = np.asarray(probabilities, dtype=float).ravel()
    if len(frame) != p.size:
        raise ValueError("frame and probabilities must align")
    values = frame[attribute].astype(str).to_numpy()

    global_threshold = float(np.quantile(p, 1.0 - capacity_fraction))
    rows: list[dict[str, object]] = []
    for group in sorted(pd.unique(values)):
        mask = values == group
        rows.append(
            {
                "attribute": attribute,
                "group": group,
                "n": int(mask.sum()),
                "group_threshold": round(float(np.quantile(p[mask], 1.0 - capacity_fraction)), 6),
                "global_threshold": round(global_threshold, 6),
            }
        )
    table = pd.DataFrame(rows)
    table["threshold_difference"] = (
        table["group_threshold"] - table["global_threshold"]
    ).round(6)
    return table
