"""Candidate models, calibration, splitting and explanation.

The interpretable baseline is kept on purpose: a clinical governance committee
approves what it can read, and if the boosted model cannot beat a transparent
logistic regression on validation average precision, the extra complexity has not
earned its place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from .features import FEATURE_CONTRACT, TARGET, FeatureContract, build_preprocessor, prepare_features

MODEL_KINDS: tuple[str, ...] = ("logistic", "gradient_boosting")


def build_model(
    kind: str = "gradient_boosting",
    contract: FeatureContract = FEATURE_CONTRACT,
    seed: int = 19,
) -> Pipeline:
    """Preprocessing plus an estimator as one fittable object."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"kind must be one of {MODEL_KINDS}, got {kind!r}")
    estimator = (
        LogisticRegression(
            class_weight="balanced", max_iter=3_000, C=0.6, solver="lbfgs", random_state=seed
        )
        if kind == "logistic"
        else HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=450,
            max_depth=5,
            min_samples_leaf=45,
            l2_regularization=1.5,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=seed,
        )
    )
    return Pipeline(steps=[("preprocess", build_preprocessor(contract)), ("model", estimator)])


def three_way_split(
    frame: pd.DataFrame, seed: int = 19, target: str = TARGET
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 60/20/20 split: fit, then choose, then report once."""
    if target not in frame.columns:
        raise ValueError(f"frame must contain the '{target}' column")
    train_part, holdout = train_test_split(
        frame, test_size=0.4, stratify=frame[target], random_state=seed
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, stratify=holdout[target], random_state=seed
    )
    return (
        train_part.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def fit_model(train_frame: pd.DataFrame, kind: str, seed: int = 19) -> Pipeline:
    """Fit one candidate on the training rows.

    Boosting gets balanced sample weights rather than resampling: oversampling
    would distort the probability scale, and calibrated probabilities are the
    point of the exercise.
    """
    features = prepare_features(train_frame)
    target = train_frame[TARGET].to_numpy(dtype=int)
    model = build_model(kind, seed=seed)
    if kind == "gradient_boosting":
        model.fit(features, target, model__sample_weight=compute_sample_weight("balanced", target))
    else:
        model.fit(features, target)
    return model


def fit_calibrated(
    train_frame: pd.DataFrame,
    kind: str = "gradient_boosting",
    method: str = "isotonic",
    folds: int = 4,
    seed: int = 19,
) -> CalibratedClassifierCV:
    """Cross-fitted calibration wrapper, fitted inside the training rows only."""
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError("method must be 'isotonic' or 'sigmoid'")
    calibrated = CalibratedClassifierCV(build_model(kind, seed=seed), method=method, cv=folds)
    calibrated.fit(prepare_features(train_frame), train_frame[TARGET].to_numpy(dtype=int))
    return calibrated


def predict_risk(model, frame: pd.DataFrame) -> np.ndarray:
    """Readmission probability for raw encounter rows."""
    return model.predict_proba(prepare_features(frame))[:, 1]


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Standardised logistic coefficients as odds ratios, strongest first."""
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise TypeError("coefficient_table requires a linear model")
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = estimator.coef_.ravel()
    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients.round(5),
            "odds_ratio": np.exp(coefficients).round(4),
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    model, frame: pd.DataFrame, seed: int = 19, n_repeats: int = 3
) -> pd.DataFrame:
    """Permutation importance scored by average precision.

    Reported next to the logistic coefficients on purpose: when a transparent
    linear model and a permutation test agree on the top drivers, the model is
    much more likely to have learned clinical signal than an artefact.
    """
    features = prepare_features(frame)
    result = permutation_importance(
        model,
        features,
        frame[TARGET].to_numpy(dtype=int),
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": list(features.columns),
            "importance_mean": result.importances_mean.round(6),
            "importance_std": result.importances_std.round(6),
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
