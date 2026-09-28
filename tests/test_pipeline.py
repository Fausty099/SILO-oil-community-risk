"""Unit tests for the SILO pipeline (synthetic data - no raw downloads needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silo.evaluate import _base_rate, _persistence, score
from silo.features import add_features
from silo.labels import add_future_conflict, apply_label, fit_threshold


@pytest.fixture
def panel() -> pd.DataFrame:
    rows = []
    for state in ["Rivers", "Delta"]:
        for i, year in enumerate(range(2010, 2020)):
            rows.append(
                {
                    "state": state,
                    "year": year,
                    "conflict_events": 10 + i * (2 if state == "Rivers" else 5),
                    "total_fatalities": 5 + i,
                    "protest_events": 2 + i,
                    "civilian_share": 0.3,
                    "flare_volume_bcm": 1.0 + 0.1 * i,
                    "flare_site_count": 10,
                }
            )
    return pd.DataFrame(rows)


def test_features_use_only_past_and_present(panel):
    """Perturbing a future year must not change an earlier year's features."""
    base = add_features(panel)
    tampered = panel.copy()
    mask = (tampered["state"] == "Rivers") & (tampered["year"] == 2019)
    tampered.loc[mask, "conflict_events"] = 9999
    after = add_features(tampered)

    earlier = base["year"] < 2019
    cols = [c for c in base.columns if c not in {"state", "year"}]
    pd.testing.assert_frame_equal(base.loc[earlier, cols], after.loc[earlier, cols])


def test_state_history_mean_excludes_current_row(panel):
    out = add_features(panel)
    rivers = out[out["state"] == "Rivers"].sort_values("year")
    assert np.isnan(rivers["state_history_mean"].iloc[0])
    assert rivers["state_history_mean"].iloc[2] == pytest.approx(
        rivers["conflict_events"].iloc[:2].mean()
    )


def test_future_conflict_is_next_year_same_state(panel):
    out = add_future_conflict(panel)
    rivers = out[out["state"] == "Rivers"].sort_values("year")
    assert rivers["future_conflict"].iloc[0] == rivers["conflict_events"].iloc[1]
    assert np.isnan(rivers["future_conflict"].iloc[-1])


def test_threshold_is_fitted_on_training_years_only(panel):
    frame = add_future_conflict(add_features(panel))
    early = fit_threshold(frame, pd.Series(range(2010, 2015)))
    full = fit_threshold(frame)
    assert early < full, "later years are more violent, so the full-panel threshold leaks"


def test_label_requires_both_conditions(panel):
    frame = add_future_conflict(add_features(panel))
    labelled = apply_label(frame, threshold=1e9)
    assert labelled["slo_erosion"].dropna().sum() == 0, "impossible threshold must never fire"

    labelled = apply_label(frame, threshold=0.0)
    escalating = labelled["future_conflict"] > 1.2 * labelled["conflict_3yr"]
    assert (labelled.loc[~escalating.fillna(False), "slo_erosion"].dropna() == 0).all()


def test_last_year_has_no_label(panel):
    frame = add_future_conflict(add_features(panel))
    labelled = apply_label(frame, fit_threshold(frame))
    assert labelled.loc[labelled["year"] == 2019, "slo_erosion"].isna().all()


def test_persistence_baseline_repeats_last_observed_label():
    train = pd.DataFrame(
        {
            "state": ["Rivers", "Rivers", "Delta", "Delta"],
            "year": [2015, 2016, 2015, 2016],
            "slo_erosion": [0, 1, 1, 0],
        }
    )
    test = pd.DataFrame({"state": ["Rivers", "Delta"], "year": [2017, 2017]})
    assert list(_persistence(train, test)) == [1.0, 0.0]
    assert list(_base_rate(train, test)) == [0.5, 0.5]


def test_baseline_falls_back_for_unseen_state():
    train = pd.DataFrame({"state": ["Rivers"], "year": [2016], "slo_erosion": [1]})
    test = pd.DataFrame({"state": ["Ondo"], "year": [2017]})
    assert _persistence(train, test)[0] == 1.0


def test_score_reports_brier_and_ranks_by_pr_auc():
    predictions = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1] * 3,
            "p": [0.1, 0.9, 0.2, 0.8] * 2 + [0.9, 0.1, 0.8, 0.2],
            "model": ["good"] * 8 + ["bad"] * 4,
        }
    )
    table = score(predictions).set_index("model")
    assert table.loc["good", "brier"] < table.loc["bad", "brier"]
    assert table.index[0] == "good"
