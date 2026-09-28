"""Shared configuration for the SILO pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
VIIRS_DIR = RAW_DIR / "viirs"
ACLED_DIR = RAW_DIR / "acled"
BOUNDARY_DIR = RAW_DIR / "boundaries"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
MODEL_DIR = ROOT / "models"

NIGER_DELTA_STATES = [
    "Rivers",
    "Delta",
    "Bayelsa",
    "Akwa Ibom",
    "Edo",
    "Imo",
    "Ondo",
]

# GADM writes state names without separators (e.g. "AkwaIbom"); ACLED uses spaces.
GADM_NAME_FIXES = {"AkwaIbom": "Akwa Ibom", "CrossRiver": "Cross River"}

# Site-level VIIRS flare volumes are published by the Earth Observation Group
# for 2012-2020 only; later years are country aggregates behind a rate-limited API.
FLARE_YEARS = list(range(2012, 2021))

# Conflict-only panel extends as far back as ACLED covers Nigeria.
CONFLICT_START_YEAR = 1997
# Last full calendar year in the ACLED export; later years are partial.
PANEL_END_YEAR = 2025

# Primary specification: flare features are only defined inside FLARE_YEARS and
# the label needs year t+1, so the modelling frame stops one year earlier.
FLARE_MODEL_YEARS = (FLARE_YEARS[0], FLARE_YEARS[-1] - 1)
# Secondary specification: conflict-only features, far more rows.
CONFLICT_MODEL_YEARS = (2000, PANEL_END_YEAR - 1)

RANDOM_STATE = 42

# Label rule parameters (see labels.py).
LABEL_PERCENTILE = 0.60
LABEL_BASELINE_MULTIPLE = 1.2


def ensure_dirs() -> None:
    for d in (
        VIIRS_DIR,
        ACLED_DIR,
        BOUNDARY_DIR,
        PROCESSED_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        MODEL_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
