"""Download every input SILO needs from its public source.

Three streams are fetched:

* ACLED Nigeria event counts by admin1/admin2 and month-year, via the
  Humanitarian Data Exchange package API (resource URLs change on every
  weekly refresh, so they are resolved at run time rather than hard coded).
* VIIRS site-level gas flare volumes (latitude, longitude, BCM) published by
  the Earth Observation Group at the Colorado School of Mines.
* GADM v4.1 Nigeria admin-1 boundaries, used to assign flare sites to states.

Files already present on disk are not re-downloaded.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .config import ACLED_DIR, BOUNDARY_DIR, VIIRS_DIR, ensure_dirs

HDX_PACKAGE = "https://data.humdata.org/api/3/action/package_show?id=nigeria-acled-conflict-data"

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_NGA_1.json"

EOG_BASE = "https://eogdata.mines.edu/global_flare_data"
EOG_FILES = {
    "2012-2016": "VIIRS_Global_flaring_d.7_slope_0.0298_2012-2016_web.xlsx",
    "2017": "VIIRS_Global_flaring_d.7_slope_0.029353_2017_web_v1.xlsx",
    "2018": "VIIRS_Global_flaring_d.7_slope_0.029353_2018_web.xlsx",
    "2019": "VIIRS_Global_flaring_d.7_slope_0.029353_2019_web_v20201114.xlsx",
    "2020": "VIIRS_Global_flaring_d.7_slope_0.029353_2020_web_v1.xlsx",
}

ACLED_STREAMS = {
    "political_violence": "political_violence_events_and_fatalities",
    "civilian_targeting": "civilian_targeting_events_and_fatalities",
    "demonstrations": "demonstration_events",
}


def _download(url: str, target: Path) -> Path:
    if target.exists():
        print(f"  cached  {target.name}")
        return target
    print(f"  fetch   {target.name}")
    with urllib.request.urlopen(url, timeout=300) as response:
        target.write_bytes(response.read())
    return target


def acled_resource_urls() -> dict[str, str]:
    """Resolve the current HDX download URL for each ACLED stream."""
    with urllib.request.urlopen(HDX_PACKAGE, timeout=120) as response:
        package = json.load(response)["result"]

    urls: dict[str, str] = {}
    for stream, marker in ACLED_STREAMS.items():
        for resource in package["resources"]:
            if marker in resource["name"].lower():
                urls[stream] = resource["url"]
                break
    missing = set(ACLED_STREAMS) - set(urls)
    if missing:
        raise RuntimeError(f"HDX package no longer exposes: {sorted(missing)}")
    return urls


def fetch_all() -> dict[str, Path]:
    ensure_dirs()
    paths: dict[str, Path] = {}

    print("ACLED (HDX)")
    for stream, url in acled_resource_urls().items():
        paths[f"acled_{stream}"] = _download(url, ACLED_DIR / f"{stream}.xlsx")

    print("VIIRS flare sites (Earth Observation Group)")
    for key, name in EOG_FILES.items():
        paths[f"viirs_{key}"] = _download(f"{EOG_BASE}/{name}", VIIRS_DIR / name)

    print("Boundaries (GADM v4.1)")
    paths["boundaries"] = _download(GADM_URL, BOUNDARY_DIR / "gadm41_NGA_1.json")

    return paths


if __name__ == "__main__":
    fetch_all()
