"""Leak-safe feature engineering on the state x year panel.

Every feature for year t is computable from information available at the end of
year t. Rolling statistics therefore always end at t, and any "historical"
statistic uses an expanding window that excludes the current row where the
current row would otherwise smuggle in its own outcome.
"""

from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR

CONFLICT_FEATURES = [
    "conflict_events",
    "conflict_lag1",
    "conflict_lag2",
    "conflict_change",
    "conflict_accel",
    "conflict_3yr",
    "conflict_anomaly",
    "fatalities",
    "fatalities_3yr",
    "fatalities_per_event",
    "protest_events",
    "protest_share",
    "civilian_share",
    "state_history_mean",
    "years_since_peak",
]

FLARE_FEATURES = [
    "flare_volume_bcm",
    "flare_lag1",
    "flare_change",
    "flare_3yr",
    "flare_anomaly",
    "flare_site_count",
    "flare_per_site",
    "flare_conflict_index",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.where(denominator > 0)).fillna(0.0)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["state", "year"]).copy()
    df = df.rename(columns={"total_fatalities": "fatalities"})
    by_state = df.groupby("state", group_keys=False)

    df["conflict_lag1"] = by_state["conflict_events"].shift(1)
    df["conflict_lag2"] = by_state["conflict_events"].shift(2)
    df["conflict_change"] = df["conflict_events"] - df["conflict_lag1"]
    df["conflict_accel"] = df["conflict_change"] - by_state["conflict_change"].shift(1)
    df["conflict_3yr"] = by_state["conflict_events"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["conflict_anomaly"] = _safe_ratio(df["conflict_events"], df["conflict_3yr"])

    df["fatalities_3yr"] = by_state["fatalities"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["fatalities_per_event"] = _safe_ratio(df["fatalities"], df["conflict_events"])
    df["protest_share"] = _safe_ratio(df["protest_events"], df["conflict_events"])

    # Expanding mean of *past* years only (shifted) - no contamination from t.
    df["state_history_mean"] = by_state["conflict_events"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    running_max = by_state["conflict_events"].transform(lambda s: s.cummax())
    peak_year = df["year"].where(df["conflict_events"].eq(running_max))
    df["years_since_peak"] = df["year"] - peak_year.groupby(df["state"]).ffill()

    df["flare_lag1"] = by_state["flare_volume_bcm"].shift(1)
    df["flare_change"] = df["flare_volume_bcm"] - df["flare_lag1"]
    df["flare_3yr"] = by_state["flare_volume_bcm"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["flare_anomaly"] = _safe_ratio(df["flare_volume_bcm"], df["flare_3yr"])
    df["flare_per_site"] = _safe_ratio(df["flare_volume_bcm"], df["flare_site_count"])
    df["flare_conflict_index"] = df["flare_anomaly"] * df["conflict_anomaly"]

    return df.reset_index(drop=True)


def build_features() -> pd.DataFrame:
    panel = pd.read_csv(PROCESSED_DIR / "silo_panel.csv")
    features = add_features(panel)
    features.to_csv(PROCESSED_DIR / "silo_features.csv", index=False)
    print(f"  features: {features.shape[0]} rows x {features.shape[1]} columns")
    return features


if __name__ == "__main__":
    build_features()
