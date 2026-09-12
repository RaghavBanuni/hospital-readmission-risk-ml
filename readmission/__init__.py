"""30-day readmission risk with capacity-aware evaluation and a fairness audit.

Modules
-------
data       synthetic discharge encounters from a documented log-odds model
features   feature contract, coercion and the preprocessing pipeline
model      candidate estimators, calibration and permutation importance
evaluate   discrimination, calibration, capacity metrics and net benefit
fairness   subgroup metrics and gap summaries
"""

from .data import CohortConfig, generate_encounters, readmission_rate_by
from .evaluate import (
    calibration_table,
    capacity_metrics,
    discrimination_metrics,
    expected_calibration_error,
    net_benefit_table,
)
from .fairness import fairness_gaps, subgroup_metrics
from .features import FEATURE_CONTRACT, prepare_features
from .model import build_model, fit_calibrated, permutation_importance_table

__all__ = [
    "CohortConfig",
    "FEATURE_CONTRACT",
    "build_model",
    "calibration_table",
    "capacity_metrics",
    "discrimination_metrics",
    "expected_calibration_error",
    "fairness_gaps",
    "fit_calibrated",
    "generate_encounters",
    "net_benefit_table",
    "permutation_importance_table",
    "prepare_features",
    "readmission_rate_by",
    "subgroup_metrics",
]

__version__ = "1.0.0"
