"""Feature contract, coercion and preprocessing.

Two decisions are encoded here, both deliberate.

**Nothing observed after discharge is a feature.** The contract lists exactly the
fields a discharge planner holds on the day the patient leaves.  Anything that
becomes known later - a post-discharge phone call, a subsequent claim - would leak
the outcome, so it cannot even be referenced.

**Sex, insurance and preferred language are audit dimensions, not predictors.**
They are excluded from the model matrix and kept for
:mod:`readmission.fairness`.  Removing them does *not* make the model fair -
``follow_up_booked`` still carries the payer disparity built into the generator -
which is precisely why the audit exists rather than a checkbox.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import age_band

TARGET: str = "readmitted_30d"
ID_COLUMN: str = "encounter_id"
AUDIT_ATTRIBUTES: tuple[str, ...] = ("age_band", "sex", "insurance", "preferred_language")


@dataclass(frozen=True)
class FeatureContract:
    """Model inputs grouped by how they must be preprocessed."""

    numeric: tuple[str, ...]
    boolean: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str = TARGET

    @property
    def all_features(self) -> list[str]:
        return [*self.numeric, *self.boolean, *self.categorical]

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.all_features if column not in frame.columns]
        if missing:
            raise ValueError(f"encounter data is missing required features: {missing}")

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "numeric": list(self.numeric),
            "boolean": list(self.boolean),
            "categorical": list(self.categorical),
            "target": [self.target],
            "excluded_audit_attributes": list(AUDIT_ATTRIBUTES),
        }


FEATURE_CONTRACT = FeatureContract(
    numeric=(
        "age",
        "prior_admissions_12m",
        "prior_ed_visits_12m",
        "days_since_last_discharge",
        "length_of_stay_days",
        "icu_hours",
        "n_procedures",
        "comorbidity_count",
        "n_medications",
        "haemoglobin_g_dl",
        "sodium_mmol_l",
        "creatinine_mg_dl",
        "albumin_g_dl",
    ),
    boolean=(
        "lives_alone",
        "diabetes",
        "heart_failure",
        "copd",
        "renal_disease",
        "polypharmacy",
        "discharge_against_advice",
        "follow_up_booked",
    ),
    categorical=("admission_type", "discharge_disposition"),
)


def with_audit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the derived ``age_band`` used by the fairness audit."""
    if "age" not in frame.columns:
        raise ValueError("frame must contain 'age' to derive age bands")
    enriched = frame.copy()
    enriched["age_band"] = age_band(enriched["age"]).to_numpy()
    return enriched


def prepare_features(
    frame: pd.DataFrame, contract: FeatureContract = FEATURE_CONTRACT
) -> pd.DataFrame:
    """Coerce raw encounter rows into the model matrix layout."""
    contract.validate(frame)
    prepared = frame.loc[:, contract.all_features].copy()
    for column in contract.numeric:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    for column in contract.boolean:
        prepared[column] = prepared[column].astype("boolean").astype("float64")
    for column in contract.categorical:
        prepared[column] = prepared[column].astype("string").fillna("unknown").astype(object)
    return prepared


def build_preprocessor(contract: FeatureContract = FEATURE_CONTRACT) -> ColumnTransformer:
    """Impute, flag missingness, scale and encode.

    ``add_indicator=True`` matters clinically: an albumin that was never ordered is
    itself informative (it usually means the clinician was not worried), and
    silently imputing the median would throw that signal away.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    binary = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(contract.numeric)),
            ("binary", binary, list(contract.boolean)),
            ("categorical", categorical, list(contract.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
