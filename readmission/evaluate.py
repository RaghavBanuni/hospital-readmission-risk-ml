"""Evaluation for a model that will be consumed under a capacity constraint.

The care-transition team can call a fixed number of patients per week, so the
questions that matter are:

* within the top *k*% of discharges by risk, how many readmissions do we reach
  (:func:`capacity_metrics`)?
* do the probabilities mean what they say (:func:`calibration_table`,
  :func:`expected_calibration_error`)?
* at which threshold is the net benefit highest given what outreach costs and what
  a readmission costs (:func:`net_benefit_table`)?

Accuracy is never reported: predicting "nobody bounces back" scores 85%.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class OutreachEconomics:
    """What the intervention costs and what it saves.

    outreach_cost:
        Cost of one transition-care contact (nurse time, scheduling).
    readmission_cost:
        Average cost of an avoidable 30-day readmission.
    prevention_rate:
        Share of readmissions that outreach actually prevents.  Published
        transitional-care effects are modest, so the default is deliberately
        conservative.
    """

    outreach_cost: float = 120.0
    readmission_cost: float = 11_000.0
    prevention_rate: float = 0.20

    def validate(self) -> None:
        if self.outreach_cost < 0 or self.readmission_cost <= 0:
            raise ValueError("outreach_cost must be >= 0 and readmission_cost > 0")
        if not 0.0 < self.prevention_rate <= 1.0:
            raise ValueError("prevention_rate must be in (0, 1]")

    @property
    def value_per_true_positive(self) -> float:
        return self.prevention_rate * self.readmission_cost

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _checked(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    p = np.asarray(probabilities, dtype=float).ravel()
    if y.size != p.size:
        raise ValueError("labels and probabilities must have the same length")
    if y.size < 20:
        raise ValueError("at least twenty encounters are required")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("probabilities must lie in [0, 1]")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be binary 0/1")
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("evaluation needs both classes present")
    return y, p


def discrimination_metrics(y_true, probabilities) -> dict[str, float]:
    """Ranking and probability quality."""
    y, p = _checked(y_true, probabilities)
    return {
        "n": int(y.size),
        "readmission_rate": round(float(y.mean()), 5),
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "average_precision": round(float(average_precision_score(y, p)), 5),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9))), 5),
    }


def calibration_table(y_true, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted risk against observed readmission rate."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y, p = _checked(y_true, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignment = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)

    rows: list[dict[str, float]] = []
    for index in range(n_bins):
        mask = assignment == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": index + 1,
                "lower": round(float(edges[index]), 3),
                "upper": round(float(edges[index + 1]), 3),
                "encounters": int(mask.sum()),
                "mean_predicted": round(float(p[mask].mean()), 5),
                "observed_rate": round(float(y[mask].mean()), 5),
                "gap": round(float(p[mask].mean() - y[mask].mean()), 5),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Volume-weighted mean absolute gap between predicted and observed rates."""
    table = calibration_table(y_true, probabilities, n_bins)
    weights = table["encounters"].to_numpy(dtype=float)
    gaps = table["gap"].abs().to_numpy(dtype=float)
    return round(float(np.average(gaps, weights=weights)), 5)


def calibration_in_the_large(y_true, probabilities) -> float:
    """Mean predicted risk minus observed rate; positive means over-prediction."""
    y, p = _checked(y_true, probabilities)
    return round(float(p.mean() - y.mean()), 5)


def threshold_for_capacity(probabilities, capacity_fraction: float = 0.08) -> float:
    """Risk cut-off that selects the top ``capacity_fraction`` of discharges."""
    if not 0.0 < capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be in (0, 1]")
    p = np.asarray(probabilities, dtype=float).ravel()
    return float(np.quantile(p, 1.0 - capacity_fraction))


def capacity_metrics(y_true, probabilities, capacity_fraction: float = 0.08) -> dict[str, float]:
    """What the top ``capacity_fraction`` of the risk list actually reaches."""
    y, p = _checked(y_true, probabilities)
    if not 0.0 < capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be in (0, 1]")

    n_contacts = max(int(round(y.size * capacity_fraction)), 1)
    ranked = np.argsort(-p, kind="mergesort")[:n_contacts]
    flagged = np.zeros(y.size, dtype=bool)
    flagged[ranked] = True

    reached = int(y[flagged].sum())
    prevalence = float(y.mean())
    precision = reached / n_contacts
    return {
        "capacity_fraction": float(capacity_fraction),
        "contacts": int(n_contacts),
        "readmissions_reached": reached,
        "precision": round(precision, 5),
        "recall": round(reached / int(y.sum()), 5),
        "lift_over_random": round(precision / prevalence if prevalence else 0.0, 3),
        "missed_readmissions": int(y.sum()) - reached,
    }


def net_benefit_table(
    y_true,
    probabilities,
    economics: OutreachEconomics | None = None,
    n_thresholds: int = 40,
) -> pd.DataFrame:
    """Monetary net benefit at every candidate risk threshold.

    ``net_benefit = true_positives * prevention_rate * readmission_cost
                    - contacts * outreach_cost``
    """
    costs = economics or OutreachEconomics()
    costs.validate()
    y, p = _checked(y_true, probabilities)

    quantiles = np.linspace(0.50, 0.999, max(n_thresholds, 2))
    thresholds = np.unique(np.round(np.quantile(p, quantiles), 6))
    total_readmissions = int(y.sum())

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        flagged = p >= threshold
        contacts = int(flagged.sum())
        if contacts == 0:
            continue
        true_positives = int(y[flagged].sum())
        benefit = true_positives * costs.value_per_true_positive
        spend = contacts * costs.outreach_cost
        rows.append(
            {
                "threshold": float(threshold),
                "contacts": contacts,
                "contact_rate": round(contacts / y.size, 5),
                "precision": round(true_positives / contacts, 5),
                "recall": round(true_positives / total_readmissions, 5),
                "expected_readmissions_prevented": round(
                    true_positives * costs.prevention_rate, 2
                ),
                "outreach_spend": round(spend, 2),
                "net_benefit": round(benefit - spend, 2),
                "net_benefit_per_contact": round((benefit - spend) / contacts, 2),
            }
        )
    if not rows:
        raise ValueError("no threshold selected any patient")
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def best_net_benefit(table: pd.DataFrame) -> dict[str, float]:
    """Highest-net-benefit row; ties break towards fewer contacts."""
    if table.empty:
        raise ValueError("net benefit table is empty")
    best = table["net_benefit"].max()
    candidates = table[table["net_benefit"] == best]
    chosen = candidates.loc[candidates["contacts"].idxmin()]
    return {key: float(value) for key, value in chosen.to_dict().items()}
