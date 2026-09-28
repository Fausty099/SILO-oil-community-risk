"""Model benchmark with naive baselines, rolling-origin evaluation and calibration.

Design notes
------------
* Evaluation is *rolling-origin* (expanding window): for every held-out year the
  model sees only strictly earlier years. A single train/test cut on 7 states x
  8 usable years leaves too few positives to say anything; pooling the
  out-of-sample predictions of several origins is the honest way to get a
  usable test set out of a short panel.
* The label threshold is refitted inside every fold (see labels.py).
* Two naive baselines are scored on exactly the same folds: persistence (repeat
  last year's label) and the state's historical base rate. Any learned model
  that does not beat both is not worth deploying.
* Probabilities are scored with Brier score and a reliability curve, because a
  risk score that is used for engagement prioritisation is only useful if the
  number it reports means something.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import RANDOM_STATE, TABLE_DIR
from .labels import apply_label, fit_threshold

warnings.filterwarnings("ignore", category=UserWarning)


def make_models() -> dict[str, Pipeline]:
    """The five algorithms named in the proposal, all as scaled pipelines."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "XGBoost": Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    SVC(
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Neural Net (MLP)": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _persistence(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline 1: this state was flagged last year, so flag it again."""
    last = (
        train.sort_values("year")
        .groupby("state")["slo_erosion"]
        .last()
        .astype(float)
    )
    return test["state"].map(last).fillna(train["slo_erosion"].mean()).to_numpy()


def _base_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline 2: the state's historical share of erosion years."""
    rate = train.groupby("state")["slo_erosion"].mean()
    return test["state"].map(rate).fillna(train["slo_erosion"].mean()).to_numpy()


BASELINES = {"Naive persistence": _persistence, "Historical base rate": _base_rate}


def rolling_origin_predictions(
    frame: pd.DataFrame, feature_cols: list[str], min_train_years: int
) -> pd.DataFrame:
    """Out-of-sample probabilities for every model on every held-out year."""
    years = sorted(frame["year"].unique())
    rows: list[pd.DataFrame] = []

    for test_year in years[min_train_years:]:
        train_years = [y for y in years if y < test_year]
        fold = apply_label(frame, fit_threshold(frame, pd.Series(train_years)))
        fold = fold.dropna(subset=["slo_erosion"] + feature_cols)
        fold["slo_erosion"] = fold["slo_erosion"].astype(int)

        train = fold[fold["year"].isin(train_years)]
        test = fold[fold["year"] == test_year]
        if test.empty or train["slo_erosion"].nunique() < 2:
            continue

        base = test[["state", "year", "slo_erosion"]].rename(
            columns={"slo_erosion": "y_true"}
        )
        for name, fn in BASELINES.items():
            rows.append(base.assign(model=name, p=fn(train, test)))

        for name, model in make_models().items():
            model.fit(train[feature_cols], train["slo_erosion"])
            proba = model.predict_proba(test[feature_cols])[:, 1]
            rows.append(base.assign(model=name, p=proba))

    return pd.concat(rows, ignore_index=True)


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, group in predictions.groupby("model", sort=False):
        y, p = group["y_true"].to_numpy(), group["p"].to_numpy()
        yhat = (p >= 0.5).astype(int)
        both_classes = len(np.unique(y)) > 1
        out.append(
            {
                "model": name,
                "n_test": len(y),
                "positives": int(y.sum()),
                "precision": precision_score(y, yhat, zero_division=0),
                "recall": recall_score(y, yhat, zero_division=0),
                "f1": f1_score(y, yhat, zero_division=0),
                "roc_auc": roc_auc_score(y, p) if both_classes else np.nan,
                "pr_auc": average_precision_score(y, p) if both_classes else np.nan,
                "brier": brier_score_loss(y, np.clip(p, 0, 1)),
            }
        )
    return pd.DataFrame(out).sort_values("pr_auc", ascending=False, na_position="last")


def reliability(predictions: pd.DataFrame, model: str, bins: int = 5) -> pd.DataFrame:
    group = predictions[predictions["model"] == model]
    observed, predicted = calibration_curve(
        group["y_true"], np.clip(group["p"], 0, 1), n_bins=bins, strategy="quantile"
    )
    return pd.DataFrame({"model": model, "mean_predicted": predicted, "observed": observed})


def run(frame: pd.DataFrame, feature_cols: list[str], tag: str, min_train_years: int):
    predictions = rolling_origin_predictions(frame, feature_cols, min_train_years)
    table = score(predictions)
    predictions.to_csv(TABLE_DIR / f"predictions_{tag}.csv", index=False)
    table.to_csv(TABLE_DIR / f"benchmark_{tag}.csv", index=False)

    curves = pd.concat(
        [reliability(predictions, m) for m in table["model"] if m in set(predictions["model"])],
        ignore_index=True,
    )
    curves.to_csv(TABLE_DIR / f"calibration_{tag}.csv", index=False)
    return predictions, table, curves
