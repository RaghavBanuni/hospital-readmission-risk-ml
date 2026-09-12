"""Tests for the cohort generator and the feature contract.

The generator's log-odds coefficients are documented, so every documented driver
is asserted here.  If a refactor breaks the link between heart failure and
readmission, the suite says so rather than the model quietly getting worse.
"""

from __future__ import annotations

import pandas as pd
import pytest

from readmission.data import CohortConfig, generate_encounters, readmission_rate_by
from readmission.features import (
    AUDIT_ATTRIBUTES,
    FEATURE_CONTRACT,
    TARGET,
    prepare_features,
    with_audit_columns,
)


def config(**overrides: object) -> CohortConfig:
    defaults: dict[str, object] = {"n_patients": 4_000, "seed": 71}
    defaults.update(overrides)
    return CohortConfig(**defaults)  # type: ignore[arg-type]


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_encounters(config()), generate_encounters(config()))


def test_readmission_rate_lands_near_the_target():
    frame = generate_encounters(config(target_rate=0.18))
    assert 0.15 < frame[TARGET].mean() < 0.21


def test_emergency_admissions_readmit_more_than_elective():
    frame = generate_encounters(config())
    rates = frame.groupby("admission_type")[TARGET].mean()
    assert rates["emergency"] > rates["elective"]


def test_comorbidity_burden_increases_risk():
    frame = generate_encounters(config())
    light = frame[frame["comorbidity_count"] <= 1][TARGET].mean()
    heavy = frame[frame["comorbidity_count"] >= 4][TARGET].mean()
    assert heavy > light


def test_prior_utilisation_increases_risk():
    frame = generate_encounters(config())
    none = frame[frame["prior_admissions_12m"] == 0][TARGET].mean()
    several = frame[frame["prior_admissions_12m"] >= 2][TARGET].mean()
    assert several > none


def test_booked_follow_up_is_protective():
    frame = generate_encounters(config())
    rates = frame.groupby("follow_up_booked")[TARGET].mean()
    assert rates[True] < rates[False]


def test_the_payer_disparity_exists_in_the_data_generating_process():
    """Medicaid and self-pay patients get fewer booked follow-ups by construction.

    This is what gives the fairness audit something real to find; an audit that can
    only ever report 'no disparity' tests nothing.
    """
    frame = generate_encounters(config())
    booking = frame.groupby("insurance")["follow_up_booked"].mean()
    assert booking["commercial"] > booking["medicaid"] > booking["self_pay"]


def test_labs_are_missing_only_where_documented():
    frame = generate_encounters(config(albumin_missing_rate=0.30, sodium_missing_rate=0.05))
    missing = frame.isna().sum()
    assert missing["albumin_g_dl"] > missing["sodium_mmol_l"] > 0
    assert missing.drop(["albumin_g_dl", "sodium_mmol_l"]).sum() == 0


def test_encounter_ids_are_unique():
    assert generate_encounters(config())["encounter_id"].is_unique


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        generate_encounters(CohortConfig(n_patients=20))
    with pytest.raises(ValueError):
        generate_encounters(CohortConfig(target_rate=0.9))


def test_rate_by_column_covers_every_level():
    frame = generate_encounters(config())
    table = readmission_rate_by(frame, "insurance")
    assert table["encounters"].sum() == len(frame)
    assert len(table) == frame["insurance"].nunique()


# --------------------------------------------------------------- feature contract
def test_sensitive_attributes_are_not_model_features():
    """Sex, insurance and language are audit dimensions, not inputs."""
    for attribute in AUDIT_ATTRIBUTES:
        assert attribute not in FEATURE_CONTRACT.all_features


def test_no_post_discharge_information_is_in_the_contract():
    forbidden = {TARGET, "encounter_id"}
    assert not forbidden & set(FEATURE_CONTRACT.all_features)


def test_prepare_features_returns_the_contract_columns(cohort):
    prepared = prepare_features(cohort)
    assert list(prepared.columns) == FEATURE_CONTRACT.all_features
    for column in FEATURE_CONTRACT.boolean:
        assert pd.api.types.is_float_dtype(prepared[column])


def test_prepare_features_keeps_lab_missingness_for_the_pipeline(cohort):
    prepared = prepare_features(cohort)
    assert prepared["albumin_g_dl"].isna().sum() == cohort["albumin_g_dl"].isna().sum()


def test_prepare_features_reports_a_missing_column(cohort):
    with pytest.raises(ValueError, match="missing required features"):
        prepare_features(cohort.drop(columns=["length_of_stay_days"]))


def test_age_bands_cover_every_patient(cohort):
    banded = with_audit_columns(cohort)
    assert banded["age_band"].notna().all()
    assert banded["age_band"].nunique() >= 4
