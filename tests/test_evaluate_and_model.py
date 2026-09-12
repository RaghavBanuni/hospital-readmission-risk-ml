"""Tests for capacity-aware evaluation and the trained model."""

from __future__ import annotations

import numpy as np
import pytest

from readmission.evaluate import (
    OutreachEconomics,
    best_net_benefit,
    calibration_in_the_large,
    calibration_table,
    capacity_metrics,
    discrimination_metrics,
    expected_calibration_error,
    net_benefit_table,
    threshold_for_capacity,
)
from readmission.model import coefficient_table, fit_model, predict_risk

PERFECT_LABELS = np.array([1] * 20 + [0] * 180)
PERFECT_RISK = PERFECT_LABELS.astype(float) * 0.9 + 0.05


def test_metrics_detect_perfect_ranking():
    metrics = discrimination_metrics(PERFECT_LABELS, PERFECT_RISK)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["readmission_rate"] == pytest.approx(0.1)


def test_metrics_require_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        discrimination_metrics(np.zeros(50, dtype=int), np.full(50, 0.1))


def test_capacity_metrics_are_hand_computable():
    result = capacity_metrics(PERFECT_LABELS, PERFECT_RISK, capacity_fraction=0.10)
    assert result["contacts"] == 20
    assert result["readmissions_reached"] == 20
    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["lift_over_random"] == pytest.approx(10.0)
    assert result["missed_readmissions"] == 0


def test_capacity_binds_when_there_are_more_cases_than_slots():
    result = capacity_metrics(PERFECT_LABELS, PERFECT_RISK, capacity_fraction=0.05)
    assert result["contacts"] == 10
    assert result["recall"] == pytest.approx(0.5)
    assert result["missed_readmissions"] == 10


def test_threshold_for_capacity_selects_the_requested_share():
    rng = np.random.default_rng(0)
    risk = rng.uniform(size=10_000)
    threshold = threshold_for_capacity(risk, 0.08)
    assert (risk >= threshold).mean() == pytest.approx(0.08, abs=0.005)


def test_invalid_capacity_is_rejected():
    with pytest.raises(ValueError):
        capacity_metrics(PERFECT_LABELS, PERFECT_RISK, capacity_fraction=0.0)


def test_calibration_of_perfect_probabilities_is_exact():
    outcomes = np.array([1] * 30 + [0] * 30)
    assert expected_calibration_error(outcomes, outcomes.astype(float)) == pytest.approx(0.0)
    assert calibration_in_the_large(outcomes, outcomes.astype(float)) == pytest.approx(0.0)


def test_over_prediction_shows_up_as_a_positive_calibration_gap():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.05, 0.30, 5_000)
    outcomes = rng.binomial(1, truth)
    inflated = np.clip(truth * 2.0, 0.0, 1.0)
    assert calibration_in_the_large(outcomes, inflated) > 0.05
    assert expected_calibration_error(outcomes, inflated) > expected_calibration_error(
        outcomes, truth
    )


def test_calibration_table_accounts_for_every_encounter():
    rng = np.random.default_rng(2)
    risk = rng.uniform(size=2_000)
    outcomes = rng.binomial(1, risk)
    table = calibration_table(outcomes, risk, n_bins=10)
    assert table["encounters"].sum() == 2_000


def test_net_benefit_matches_the_formula():
    economics = OutreachEconomics(
        outreach_cost=100.0, readmission_cost=10_000.0, prevention_rate=0.25
    )
    table = net_benefit_table(PERFECT_LABELS, PERFECT_RISK, economics)
    best = best_net_benefit(table)
    # contacting exactly the 20 readmissions: 20 * 0.25 * 10000 - 20 * 100 = 48000
    assert best["contacts"] == 20
    assert best["net_benefit"] == pytest.approx(48_000.0)


def test_expensive_outreach_narrows_the_list():
    cheap = best_net_benefit(
        net_benefit_table(
            PERFECT_LABELS,
            PERFECT_RISK,
            OutreachEconomics(outreach_cost=50.0, readmission_cost=10_000.0, prevention_rate=0.2),
        )
    )
    dear = best_net_benefit(
        net_benefit_table(
            PERFECT_LABELS,
            PERFECT_RISK,
            OutreachEconomics(outreach_cost=1_500.0, readmission_cost=10_000.0, prevention_rate=0.2),
        )
    )
    assert dear["contacts"] <= cheap["contacts"]
    assert dear["net_benefit"] < cheap["net_benefit"]


def test_invalid_economics_are_rejected():
    with pytest.raises(ValueError):
        OutreachEconomics(outreach_cost=-1.0).validate()
    with pytest.raises(ValueError):
        OutreachEconomics(prevention_rate=0.0).validate()


# ------------------------------------------------------------------- the model
def test_model_beats_chance_on_held_out_encounters(scored_test):
    _test_frame, target, risk = scored_test
    metrics = discrimination_metrics(target, risk)
    assert metrics["roc_auc"] > 0.60
    assert metrics["average_precision"] > metrics["readmission_rate"]


def test_risk_scores_are_probabilities(scored_test):
    _test_frame, _target, risk = scored_test
    assert risk.min() >= 0.0 and risk.max() <= 1.0


def test_model_handles_encounters_with_missing_labs(splits, fitted):
    _train, _validation, test = splits
    incomplete = test[test["albumin_g_dl"].isna()]
    assert len(incomplete) > 0
    assert predict_risk(fitted, incomplete).min() >= 0.0


def test_logistic_coefficients_follow_clinical_expectation(fitted):
    coefficients = coefficient_table(fitted).set_index("feature")["coefficient"]
    assert coefficients["prior_admissions_12m"] > 0
    assert coefficients["comorbidity_count"] > 0
    assert coefficients["follow_up_booked"] < 0


def test_coefficients_are_refused_for_tree_models(splits):
    train, _validation, _test = splits
    booster = fit_model(train, "gradient_boosting", seed=5)
    with pytest.raises(TypeError):
        coefficient_table(booster)


def test_unknown_model_kind_is_rejected(splits):
    train, _validation, _test = splits
    with pytest.raises(ValueError, match="kind must be one of"):
        fit_model(train, "random_forest")
