"""Synthetic discharge encounters with a transparent readmission mechanism.

Everything the model is asked to learn is written down here as log-odds
coefficients, which buys three things:

* the tests can assert that each documented driver points the right way;
* the README can describe the data without hand-waving;
* the fairness audit has something real to find - one insurance group has a lower
  follow-up booking rate, so a disparity exists in the data-generating process
  rather than being asserted in prose.

Nothing here is derived from real patients.  No PHI, no redistributed dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEXES: tuple[str, ...] = ("female", "male")
INSURANCE: tuple[str, ...] = ("commercial", "medicare", "medicaid", "self_pay")
LANGUAGES: tuple[str, ...] = ("english", "spanish", "hindi", "other")
ADMISSION_TYPES: tuple[str, ...] = ("emergency", "urgent", "elective")
DISPOSITIONS: tuple[str, ...] = ("home", "home_health", "skilled_nursing")

# Log-odds contributions of the data-generating process.
ADMISSION_TYPE_EFFECT: dict[str, float] = {
    "emergency": 0.42,
    "urgent": 0.18,
    "elective": -0.30,
}
DISPOSITION_EFFECT: dict[str, float] = {
    "home": -0.20,
    "home_health": 0.15,
    "skilled_nursing": 0.45,
}
# Follow-up booking probability differs by payer; this is the disparity the
# fairness audit is designed to surface rather than to assume away.
FOLLOW_UP_PROBABILITY: dict[str, float] = {
    "commercial": 0.82,
    "medicare": 0.74,
    "medicaid": 0.55,
    "self_pay": 0.41,
}


@dataclass(frozen=True)
class CohortConfig:
    """Cohort settings.

    n_patients:
        Number of discharge encounters.
    target_rate:
        Approximate 30-day readmission rate; the intercept is solved for.
    albumin_missing_rate / sodium_missing_rate:
        Missingness of the discharge labs - albumin is ordered far less often,
        which is why the pipeline needs imputation and missingness indicators.
    """

    n_patients: int = 30_000
    target_rate: float = 0.15
    albumin_missing_rate: float = 0.28
    sodium_missing_rate: float = 0.04
    seed: int = 47


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _solve_intercept(log_odds: np.ndarray, target_rate: float) -> float:
    low, high = -14.0, 14.0
    for _ in range(90):
        middle = (low + high) / 2.0
        if _sigmoid(log_odds + middle).mean() < target_rate:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def generate_encounters(config: CohortConfig | None = None) -> pd.DataFrame:
    """Return one row per discharge with a ``readmitted_30d`` label."""
    cfg = config or CohortConfig()
    if cfg.n_patients < 200:
        raise ValueError("n_patients must be at least 200")
    if not 0.02 <= cfg.target_rate <= 0.50:
        raise ValueError("target_rate must be between 0.02 and 0.50")
    for rate in (cfg.albumin_missing_rate, cfg.sodium_missing_rate):
        if not 0.0 <= rate < 0.8:
            raise ValueError("lab missingness rates must be in [0, 0.8)")

    rng = np.random.default_rng(cfg.seed)
    size = cfg.n_patients

    age = np.clip(rng.normal(64.0, 17.0, size), 18.0, 98.0).round(0)
    sex = rng.choice(SEXES, size=size, p=(0.51, 0.49))
    # payer mix correlates with age: Medicare skews old, commercial younger
    medicare_propensity = _sigmoid((age - 65.0) / 4.0)
    insurance = np.where(
        rng.random(size) < medicare_propensity,
        "medicare",
        rng.choice(("commercial", "medicaid", "self_pay"), size=size, p=(0.62, 0.30, 0.08)),
    )
    language = rng.choice(LANGUAGES, size=size, p=(0.72, 0.14, 0.08, 0.06))
    lives_alone = rng.random(size) < (0.18 + 0.25 * (age > 75))

    admission_type = rng.choice(ADMISSION_TYPES, size=size, p=(0.52, 0.24, 0.24))
    prior_admissions = rng.poisson(0.55 + 0.02 * np.maximum(age - 55.0, 0.0) / 5.0, size)
    prior_ed_visits = rng.poisson(0.7 + 0.35 * prior_admissions, size)
    days_since_last_discharge = np.where(
        prior_admissions > 0, rng.gamma(2.0, 70.0, size).round(0), 3_650.0
    )

    comorbidity_count = rng.binomial(9, _sigmoid((age - 60.0) / 22.0) * 0.55, size)
    diabetes = rng.random(size) < (0.16 + 0.03 * comorbidity_count)
    heart_failure = rng.random(size) < (0.07 + 0.035 * comorbidity_count)
    copd = rng.random(size) < (0.06 + 0.028 * comorbidity_count)
    renal_disease = rng.random(size) < (0.05 + 0.03 * comorbidity_count)

    length_of_stay = np.clip(
        rng.gamma(2.0, 1.7 + 0.35 * comorbidity_count, size), 0.5, 45.0
    ).round(1)
    icu_hours = np.where(
        rng.random(size) < (0.12 + 0.05 * (admission_type == "emergency")),
        np.clip(rng.gamma(2.0, 22.0, size), 1.0, 400.0).round(1),
        0.0,
    )
    n_procedures = rng.poisson(0.9 + 0.25 * (admission_type == "elective"), size)
    n_medications = np.clip(
        rng.poisson(6.0 + 1.15 * comorbidity_count, size), 0, 40
    )
    polypharmacy = n_medications >= 10

    haemoglobin = np.clip(rng.normal(12.3 - 0.35 * renal_disease, 1.7, size), 6.0, 18.0).round(1)
    sodium = np.clip(rng.normal(138.5, 3.6, size), 118.0, 152.0).round(1)
    creatinine = np.clip(
        rng.lognormal(np.log(1.0 + 0.6 * renal_disease), 0.35, size), 0.3, 9.0
    ).round(2)
    albumin = np.clip(rng.normal(3.7 - 0.25 * (comorbidity_count > 3), 0.5, size), 1.5, 5.5).round(2)

    discharge_against_advice = rng.random(size) < 0.02
    follow_up_probability = np.array([FOLLOW_UP_PROBABILITY[value] for value in insurance])
    follow_up_booked = rng.random(size) < np.where(
        discharge_against_advice, follow_up_probability * 0.3, follow_up_probability
    )
    disposition = np.where(
        (age > 78) & (length_of_stay > 6),
        rng.choice(DISPOSITIONS, size=size, p=(0.35, 0.30, 0.35)),
        rng.choice(DISPOSITIONS, size=size, p=(0.74, 0.20, 0.06)),
    )

    admission_term = np.array([ADMISSION_TYPE_EFFECT[value] for value in admission_type])
    disposition_term = np.array([DISPOSITION_EFFECT[value] for value in disposition])

    log_odds = (
        admission_term
        + disposition_term
        + 0.020 * (age - 60.0)
        + 0.30 * prior_admissions
        + 0.10 * prior_ed_visits
        + 0.16 * comorbidity_count
        + 0.35 * heart_failure
        + 0.28 * copd
        + 0.32 * renal_disease
        + 0.18 * diabetes
        + 0.05 * length_of_stay
        + 0.0025 * icu_hours
        + 0.22 * polypharmacy
        + 0.26 * lives_alone
        - 0.55 * follow_up_booked
        # a heart-failure patient with no booked follow-up is the classic bounce-back
        - 0.40 * (follow_up_booked & heart_failure)
        + 0.60 * discharge_against_advice
        - 0.16 * (haemoglobin - 12.3)
        + 0.20 * np.log(creatinine)
        - 0.30 * (albumin - 3.7)
        - 0.0004 * np.minimum(days_since_last_discharge, 365.0)
        + rng.normal(0.0, 0.40, size)  # unobserved clinical heterogeneity
    )
    log_odds += _solve_intercept(log_odds, cfg.target_rate)
    readmitted = (rng.random(size) < _sigmoid(log_odds)).astype(int)

    frame = pd.DataFrame(
        {
            "encounter_id": [f"E{index:07d}" for index in range(size)],
            "age": age,
            "sex": sex,
            "insurance": insurance,
            "preferred_language": language,
            "lives_alone": lives_alone,
            "admission_type": admission_type,
            "prior_admissions_12m": prior_admissions.astype(int),
            "prior_ed_visits_12m": prior_ed_visits.astype(int),
            "days_since_last_discharge": days_since_last_discharge,
            "length_of_stay_days": length_of_stay,
            "icu_hours": icu_hours,
            "n_procedures": n_procedures.astype(int),
            "comorbidity_count": comorbidity_count.astype(int),
            "diabetes": diabetes,
            "heart_failure": heart_failure,
            "copd": copd,
            "renal_disease": renal_disease,
            "n_medications": n_medications.astype(int),
            "polypharmacy": polypharmacy,
            "haemoglobin_g_dl": haemoglobin,
            "sodium_mmol_l": sodium,
            "creatinine_mg_dl": creatinine,
            "albumin_g_dl": albumin,
            "discharge_disposition": disposition,
            "discharge_against_advice": discharge_against_advice,
            "follow_up_booked": follow_up_booked,
            "readmitted_30d": readmitted,
        }
    )

    # Labs are ordered selectively in practice; albumin far less often than sodium.
    for column, rate in (
        ("albumin_g_dl", cfg.albumin_missing_rate),
        ("sodium_mmol_l", cfg.sodium_missing_rate),
    ):
        if rate > 0:
            frame.loc[rng.random(size) < rate, column] = np.nan

    return frame


def readmission_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Readmission rate and volume per level of a column."""
    if column not in frame.columns:
        raise ValueError(f"unknown column: {column}")
    grouped = frame.groupby(column, dropna=False)["readmitted_30d"]
    table = pd.DataFrame(
        {"encounters": grouped.size(), "readmission_rate": grouped.mean().round(4)}
    ).reset_index()
    return table.sort_values("readmission_rate", ascending=False).reset_index(drop=True)


def age_band(age: pd.Series | np.ndarray) -> pd.Series:
    """Clinical age bands used as an audit dimension."""
    values = pd.Series(np.asarray(age, dtype=float))
    return pd.cut(
        values,
        bins=[17, 40, 55, 65, 75, 85, 120],
        labels=["18-40", "41-55", "56-65", "66-75", "76-85", "86+"],
        right=True,
    ).astype(str)
