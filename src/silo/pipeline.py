"""End-to-end SILO pipeline: panel -> features -> labels -> benchmark -> figures.

Run with `python -m silo.pipeline` (see README). Raw inputs are fetched by
`python -m silo.acquire` first.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    CONFLICT_MODEL_YEARS,
    FIGURE_DIR,
    FLARE_MODEL_YEARS,
    MODEL_DIR,
    PROCESSED_DIR,
    TABLE_DIR,
    ensure_dirs,
)
from .evaluate import make_models, run
from .features import CONFLICT_FEATURES, FLARE_FEATURES, add_features
from .labels import add_future_conflict, apply_label, fit_threshold
from .panel import build_panel

SPECS = {
    "flare": {
        "years": FLARE_MODEL_YEARS,
        "features": CONFLICT_FEATURES + FLARE_FEATURES,
        "min_train_years": 4,
        "title": "Conflict + satellite flaring (2012-2019)",
    },
    "conflict_only": {
        "years": CONFLICT_MODEL_YEARS,
        "features": CONFLICT_FEATURES,
        "min_train_years": 8,
        "title": "Conflict history only (2000-2024)",
    },
}


def prepare() -> pd.DataFrame:
    panel_path = PROCESSED_DIR / "silo_panel.csv"
    panel = pd.read_csv(panel_path) if panel_path.exists() else build_panel()
    frame = add_future_conflict(add_features(panel))
    frame.to_csv(PROCESSED_DIR / "silo_features.csv", index=False)
    return frame


def plot_calibration(curves: pd.DataFrame, tag: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for model, group in curves.groupby("model", sort=False):
        ax.plot(group["mean_predicted"], group["observed"], "o-", label=model)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of SLO erosion")
    ax.set_title(f"Reliability curve - {title}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"calibration_{tag}.png", dpi=150)
    plt.close(fig)


def plot_benchmark(table: pd.DataFrame, tag: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ordered = table.sort_values("pr_auc", na_position="first")
    axes[0].barh(ordered["model"], ordered["pr_auc"].fillna(0))
    axes[0].set_xlabel("PR-AUC (higher is better)")
    ordered_brier = table.sort_values("brier", ascending=False)
    axes[1].barh(ordered_brier["model"], ordered_brier["brier"], color="indianred")
    axes[1].set_xlabel("Brier score (lower is better)")
    fig.suptitle(f"Rolling-origin benchmark - {title}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"benchmark_{tag}.png", dpi=150)
    plt.close(fig)


def fit_final_model(frame: pd.DataFrame, features: list[str]):
    """Refit XGBoost on the whole modelling window for interpretation/export."""
    labelled = apply_label(frame, fit_threshold(frame))
    labelled = labelled.dropna(subset=["slo_erosion"] + features)
    labelled["slo_erosion"] = labelled["slo_erosion"].astype(int)
    model = make_models()["XGBoost"]
    model.fit(labelled[features], labelled["slo_erosion"])
    return model, labelled


def shap_summary(model, labelled: pd.DataFrame, features: list[str], tag: str) -> None:
    import shap
    import xgboost as xgb

    # XGBoost's native TreeSHAP avoids shap's model-format parsing, which breaks
    # across xgboost major versions. Last column is the expected value.
    booster = model.named_steps["clf"].get_booster()
    matrix = xgb.DMatrix(labelled[features])
    values = booster.predict(matrix, pred_contribs=True)[:, :-1]

    shap.summary_plot(values, labelled[features], show=False, max_display=15)
    plt.title(f"SHAP feature importance ({tag})")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"shap_{tag}.png", dpi=150)
    plt.close()

    importance = (
        pd.DataFrame({"feature": features, "mean_abs_shap": abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(TABLE_DIR / f"shap_importance_{tag}.csv", index=False)


def main() -> None:
    ensure_dirs()
    frame = prepare()
    summary: dict[str, dict] = {}

    for tag, spec in SPECS.items():
        lo, hi = spec["years"]
        window = frame[frame["year"].between(lo, hi)].copy()
        features = [c for c in spec["features"] if c in window.columns]
        window = window.dropna(subset=features + ["future_conflict"])

        _, table, curves = run(window, features, tag, spec["min_train_years"])
        plot_calibration(curves, tag, spec["title"])
        plot_benchmark(table, tag, spec["title"])

        print(f"\n=== {spec['title']} ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        summary[tag] = {
            "title": spec["title"],
            "rows": len(window),
            "test_rows": int(table["n_test"].iloc[0]),
            "positives": int(table["positives"].iloc[0]),
            "best_by_pr_auc": table.iloc[0]["model"],
            "benchmark": table.to_dict(orient="records"),
        }

        if tag == "flare":
            model, labelled = fit_final_model(window, features)
            model_path = MODEL_DIR / "silo_xgboost.json"
            model.named_steps["clf"].save_model(model_path)
            shap_summary(model, labelled, features, tag)
            summary[tag]["model_path"] = str(model_path.relative_to(MODEL_DIR.parent))

    (TABLE_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote tables to {TABLE_DIR} and figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
