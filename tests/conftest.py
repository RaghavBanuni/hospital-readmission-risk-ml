"""Shared fixtures: one small cohort and one trained model per session."""

from __future__ import annotations

import pytest

from readmission.data import CohortConfig, generate_encounters
from readmission.features import TARGET, with_audit_columns
from readmission.model import fit_model, predict_risk, three_way_split


@pytest.fixture(scope="session")
def cohort():
    return with_audit_columns(
        generate_encounters(CohortConfig(n_patients=4_000, seed=303))
    )


@pytest.fixture(scope="session")
def splits(cohort):
    return three_way_split(cohort, seed=5)


@pytest.fixture(scope="session")
def fitted(splits):
    train, _validation, _test = splits
    return fit_model(train, "logistic", seed=5)


@pytest.fixture(scope="session")
def scored_test(fitted, splits):
    _train, _validation, test = splits
    return test, test[TARGET].to_numpy(dtype=int), predict_risk(fitted, test)
