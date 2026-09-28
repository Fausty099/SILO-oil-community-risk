"""SLO-erosion proxy label.

A state-year is labelled 1 when next year's conflict count is both

* high relative to the cross-sectional distribution (a percentile threshold), and
* a real escalation against the state's own recent history.

The percentile threshold is a *parameter estimated from data*, so it is fitted
on the training years only and then applied unchanged to later years. Fitting it
on the full panel - as the original notebook did - leaks the test period's
outcome distribution into the label definition.
"""

from __future__ import annotations

import pandas as pd

from .config import LABEL_BASELINE_MULTIPLE, LABEL_PERCENTILE


def add_future_conflict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["state", "year"]).copy()
    df["future_conflict"] = df.groupby("state")["conflict_events"].shift(-1)
    return df


def fit_threshold(df: pd.DataFrame, train_years: pd.Series | None = None) -> float:
    """Percentile of next-year conflict, estimated on training rows only."""
    frame = df if train_years is None else df[df["year"].isin(train_years)]
    return float(frame["future_conflict"].quantile(LABEL_PERCENTILE))


def apply_label(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["slo_erosion"] = (
        (df["future_conflict"] > threshold)
        & (df["future_conflict"] > LABEL_BASELINE_MULTIPLE * df["conflict_3yr"])
    ).astype(int)
    df.loc[df["future_conflict"].isna(), "slo_erosion"] = pd.NA
    return df
