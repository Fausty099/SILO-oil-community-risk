"""Build the state x year panel from raw ACLED, VIIRS and GADM inputs.

The flare stream is the part that differs most from the original notebook:
state flare volume is obtained by a point-in-polygon join of individual VIIRS
flare sites onto GADM state polygons, not by apportioning the national total
with fixed per-state shares.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .config import (
    ACLED_DIR,
    BOUNDARY_DIR,
    CONFLICT_START_YEAR,
    FLARE_YEARS,
    GADM_NAME_FIXES,
    NIGER_DELTA_STATES,
    PANEL_END_YEAR,
    PROCESSED_DIR,
    VIIRS_DIR,
    ensure_dirs,
)

# Sheets holding country aggregates rather than individual sites, plus the
# 2019 "all flares" sheet which duplicates the upstream/downstream sheets.
_AGGREGATE_SHEETS = ("iso_", "countries", "world", "definitions", "all flares")


def load_boundaries() -> gpd.GeoDataFrame:
    states = gpd.read_file(BOUNDARY_DIR / "gadm41_NGA_1.json")[["NAME_1", "geometry"]]
    states["state"] = states["NAME_1"].replace(GADM_NAME_FIXES)
    return states[["state", "geometry"]]


def _site_sheets(book: pd.ExcelFile) -> list[str]:
    return [
        sheet
        for sheet in book.sheet_names
        if not any(sheet.lower().startswith(p) or sheet.lower() == p for p in _AGGREGATE_SHEETS)
    ]


def load_flare_sites() -> pd.DataFrame:
    """Return one row per (flare site, year) for Nigeria, with BCM volumes."""
    frames: list[pd.DataFrame] = []

    for path in sorted(VIIRS_DIR.glob("VIIRS_Global_flaring*.xlsx")):
        book = pd.ExcelFile(path)
        for sheet in _site_sheets(book):
            raw = pd.read_excel(book, sheet)
            columns = {str(c).strip().lower(): c for c in raw.columns}
            iso_col = columns.get("iso_code") or columns.get("iso code")
            lat_col = columns.get("latitude")
            lon_col = columns.get("longitude")
            if iso_col is None or lat_col is None or lon_col is None:
                continue

            nigeria = raw[raw[iso_col] == "NGA"]
            for key, column in columns.items():
                if not key.startswith("bcm"):
                    continue
                year = int("".join(ch for ch in key if ch.isdigit()))
                sites = nigeria[[lat_col, lon_col, column]].copy()
                sites.columns = ["latitude", "longitude", "flare_bcm"]
                sites["year"] = year
                frames.append(sites)

    sites = pd.concat(frames, ignore_index=True)
    sites["flare_bcm"] = pd.to_numeric(sites["flare_bcm"], errors="coerce").fillna(0.0)
    sites = sites.dropna(subset=["latitude", "longitude"])
    return sites[sites["year"].isin(FLARE_YEARS)].reset_index(drop=True)


def flare_by_state(sites: pd.DataFrame, states: gpd.GeoDataFrame) -> pd.DataFrame:
    """Point-in-polygon join of flare sites onto states."""
    points = gpd.GeoDataFrame(
        sites,
        geometry=[Point(xy) for xy in zip(sites["longitude"], sites["latitude"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, states, how="left", predicate="within")

    unmatched = joined["state"].isna().sum()
    if unmatched:
        print(f"  {unmatched} flare site-years fell outside all state polygons (offshore); dropped")

    joined = joined.dropna(subset=["state"])
    agg = (
        joined.groupby(["state", "year"])
        .agg(flare_volume_bcm=("flare_bcm", "sum"), flare_site_count=("flare_bcm", "size"))
        .reset_index()
    )
    return agg[agg["state"].isin(NIGER_DELTA_STATES)]


def load_conflict() -> pd.DataFrame:
    """Aggregate the three ACLED event streams to state x year."""
    political = pd.read_excel(ACLED_DIR / "political_violence.xlsx", sheet_name="Data")
    civilian = pd.read_excel(ACLED_DIR / "civilian_targeting.xlsx", sheet_name="Data")
    demos = pd.read_excel(ACLED_DIR / "demonstrations.xlsx", sheet_name="Data")

    def to_state_year(frame: pd.DataFrame, events: str, fatalities: str | None) -> pd.DataFrame:
        frame = frame.rename(columns={"Admin1": "state", "Year": "year"})
        frame = frame[frame["state"].isin(NIGER_DELTA_STATES)]
        spec = {events: ("Events", "sum")}
        if fatalities:
            spec[fatalities] = ("Fatalities", "sum")
        return frame.groupby(["state", "year"]).agg(**spec).reset_index()

    pv = to_state_year(political, "violence_events", "violence_fatalities")
    ct = to_state_year(civilian, "civilian_targeting_events", "civilian_fatalities")
    dm = to_state_year(demos, "protest_events", None)

    conflict = pv.merge(ct, on=["state", "year"], how="outer").merge(
        dm, on=["state", "year"], how="outer"
    )
    conflict = conflict.fillna(0.0)

    # Civilian-targeting events are a subset of political violence events, so
    # they are kept as an intensity indicator and never added to the total.
    conflict["conflict_events"] = conflict["violence_events"] + conflict["protest_events"]
    conflict["total_fatalities"] = conflict["violence_fatalities"]
    conflict["civilian_share"] = conflict["civilian_targeting_events"] / conflict[
        "violence_events"
    ].where(conflict["violence_events"] > 0)
    conflict["civilian_share"] = conflict["civilian_share"].fillna(0.0)

    keep = conflict["year"].between(CONFLICT_START_YEAR, PANEL_END_YEAR)
    return conflict[keep].reset_index(drop=True)


def build_panel() -> pd.DataFrame:
    ensure_dirs()
    print("Building panel")
    states = load_boundaries()
    conflict = load_conflict()
    flare = flare_by_state(load_flare_sites(), states)

    years = range(int(conflict["year"].min()), int(conflict["year"].max()) + 1)
    grid = pd.DataFrame(
        [{"state": s, "year": y} for s in NIGER_DELTA_STATES for y in years]
    )

    panel = grid.merge(conflict, on=["state", "year"], how="left")
    count_cols = [
        "violence_events",
        "violence_fatalities",
        "civilian_targeting_events",
        "civilian_fatalities",
        "protest_events",
        "conflict_events",
        "total_fatalities",
        "civilian_share",
    ]
    panel[count_cols] = panel[count_cols].fillna(0.0)

    # Flare volumes stay NaN outside the VIIRS site-level coverage window so
    # that downstream code can distinguish "no flaring" from "not measured".
    panel = panel.merge(flare, on=["state", "year"], how="left")
    in_coverage = panel["year"].isin(FLARE_YEARS)
    panel.loc[in_coverage, ["flare_volume_bcm", "flare_site_count"]] = panel.loc[
        in_coverage, ["flare_volume_bcm", "flare_site_count"]
    ].fillna(0.0)

    panel = panel.sort_values(["state", "year"]).reset_index(drop=True)
    panel.to_csv(PROCESSED_DIR / "silo_panel.csv", index=False)
    print(f"  panel: {panel.shape[0]} state-years, {panel['year'].min()}-{panel['year'].max()}")
    print(f"  flare coverage: {FLARE_YEARS[0]}-{FLARE_YEARS[-1]}")
    return panel


if __name__ == "__main__":
    build_panel()
