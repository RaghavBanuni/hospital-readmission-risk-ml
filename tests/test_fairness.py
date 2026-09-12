"""Tests for the fairness audit.

The core cases use a hand-built cohort with a deliberate, exactly known disparity:
two groups of one hundred patients each, same prevalence, but the model ranks one
group's cases above the threshold and the other's below it.  The equal-opportunity
gap must therefore be exactly 1.0 - if the arithmetic drifts, this test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from readmission.fairness import (
    fairness_gaps,
    subgroup_metrics,
    threshold_per_group,
    worst_equal_opportunity_gap,
)

THRESHOLD = 0.5


def biased_cohort() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Group A's cases are all flagged; group B's cases are all missed."""
    labels, risk, groups = [], [], []
    for group, positive_risk in (("group_a", 0.90), ("group_b", 0.20)):
        labels.extend([1] * 20 + [0] * 80)
        risk.extend([positive_risk] * 20 + [0.10] * 80)
        groups.extend([group] * 100)
    frame = pd.DataFrame({"payer": groups, "sex": ["female", "male"] * 100})
    return frame, np.array(labels), np.array(risk)


def test_subgroup_table_has_one_row_per_group():
    frame, labels, risk = biased_cohort()
    metrics = subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("payer", "sex"))
    assert len(metrics) == 4
    assert set(metrics["attribute"]) == {"payer", "sex"}
    assert metrics["n"].sum() == 2 * len(frame)  # each attribute partitions the cohort


def test_group_metrics_are_exact_on_the_hand_built_cohort():
    frame, labels, risk = biased_cohort()
    metrics = subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("payer",))
    indexed = metrics.set_index("group")
    assert indexed.loc["group_a", "true_positive_rate"] == pytest.approx(1.0)
    assert indexed.loc["group_a", "false_negative_rate"] == pytest.approx(0.0)
    assert indexed.loc["group_a", "selection_rate"] == pytest.approx(0.20)
    assert indexed.loc["group_b", "true_positive_rate"] == pytest.approx(0.0)
    assert indexed.loc["group_b", "false_negative_rate"] == pytest.approx(1.0)
    assert indexed.loc["group_b", "selection_rate"] == pytest.approx(0.0)
    # prevalence is identical, so the disparity is the model's, not the cohort's
    assert indexed["prevalence"].nunique() == 1


def test_equal_opportunity_gap_is_reported_exactly():
    frame, labels, risk = biased_cohort()
    metrics = subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("payer",))
    gaps = fairness_gaps(metrics)
    equal_opportunity = gaps[
        (gaps["attribute"] == "payer") & (gaps["metric"] == "false_negative_rate")
    ].iloc[0]
    assert equal_opportunity["gap"] == pytest.approx(1.0)
    assert equal_opportunity["worst_group"] == "group_b"
    assert equal_opportunity["best_group"] == "group_a"


def test_demographic_parity_gap_is_reported():
    frame, labels, risk = biased_cohort()
    gaps = fairness_gaps(
        subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("payer",))
    )
    parity = gaps[gaps["metric"] == "selection_rate"].iloc[0]
    assert parity["gap"] == pytest.approx(0.20)


def test_worst_gap_helper_finds_the_biased_attribute():
    frame, labels, risk = biased_cohort()
    gaps = fairness_gaps(
        subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("payer", "sex"))
    )
    worst = worst_equal_opportunity_gap(gaps)
    assert worst["attribute"] == "payer"
    assert worst["false_negative_rate_gap"] == pytest.approx(1.0)


def test_a_fair_model_shows_no_gap():
    frame = pd.DataFrame({"payer": ["a"] * 100 + ["b"] * 100})
    labels = np.array(([1] * 20 + [0] * 80) * 2)
    risk = np.array(([0.9] * 20 + [0.1] * 80) * 2)
    gaps = fairness_gaps(subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("payer",)))
    assert gaps[gaps["metric"] == "false_negative_rate"].iloc[0]["gap"] == pytest.approx(0.0)
    assert gaps[gaps["metric"] == "selection_rate"].iloc[0]["gap"] == pytest.approx(0.0)


def test_small_groups_are_reported_but_excluded_from_gaps():
    frame, labels, risk = biased_cohort()
    metrics = subgroup_metrics(
        frame, labels, risk, THRESHOLD, attributes=("payer",), min_group_size=150
    )
    assert len(metrics) == 2
    assert not metrics["included_in_gaps"].any()
    with pytest.raises(ValueError, match="large enough"):
        fairness_gaps(metrics)


def test_metrics_are_nan_when_a_group_has_no_cases():
    frame = pd.DataFrame({"payer": ["a"] * 100 + ["b"] * 100})
    labels = np.array([1] * 20 + [0] * 80 + [0] * 100)  # group b has no readmissions
    risk = np.array([0.9] * 20 + [0.1] * 80 + [0.1] * 100)
    metrics = subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("payer",)).set_index(
        "group"
    )
    assert np.isnan(metrics.loc["b", "true_positive_rate"])
    assert np.isnan(metrics.loc["b", "roc_auc"])
    assert metrics.loc["b", "selection_rate"] == pytest.approx(0.0)


def test_misaligned_inputs_are_rejected():
    frame, labels, risk = biased_cohort()
    with pytest.raises(ValueError, match="align"):
        subgroup_metrics(frame.head(50), labels, risk, THRESHOLD, attributes=("payer",))


def test_unknown_audit_attribute_is_rejected():
    frame, labels, risk = biased_cohort()
    with pytest.raises(ValueError, match="not present"):
        subgroup_metrics(frame, labels, risk, THRESHOLD, attributes=("ethnicity",))


def test_group_thresholds_equalise_the_selection_rate():
    frame, _labels, risk = biased_cohort()
    table = threshold_per_group(frame, risk, "payer", capacity_fraction=0.20)
    assert len(table) == 2
    # group_b's cases sit lower on the risk scale, so its threshold must drop
    indexed = table.set_index("group")
    assert indexed.loc["group_b", "group_threshold"] < indexed.loc["group_a", "group_threshold"]
    assert "threshold_difference" in table.columns


def test_audit_runs_on_the_real_model(scored_test):
    test_frame, target, risk = scored_test
    threshold = float(np.quantile(risk, 0.92))
    metrics = subgroup_metrics(
        test_frame, target, risk, threshold, attributes=("insurance", "sex"), min_group_size=50
    )
    assert metrics["included_in_gaps"].any()
    gaps = fairness_gaps(metrics)
    assert not gaps.empty
    assert gaps["gap"].min() >= 0.0
