"""Shared NBI loading/cleaning/geolocation-decoding helpers.

Extracted from notebooks/eda.ipynb (Sections 2, 3, 4.3) so the same logic is defined once
instead of being copy-pasted across the EDA notebook, the model training notebook, and the
current-risk snapshot script.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

STATES = ["CA", "FL", "MD", "MI", "NY", "TX", "WA"]
YEARS = [f"{y:02d}" for y in range(10, 26)]  # 2010 to 2025
KEY = "STRUCTURE_NUMBER_008"

FEATURE_COLS = [
    KEY, "COUNTY_CODE_003", "YEAR_BUILT_027", "ADT_029", "YEAR_ADT_030",
    "PERCENT_ADT_TRUCK_109", "STRUCTURE_KIND_043A", "STRUCTURE_TYPE_043B",
    "MAIN_UNIT_SPANS_045", "APPR_SPANS_046", "STRUCTURE_LEN_MT_049", "MAX_SPAN_LEN_MT_048",
    "DECK_WIDTH_MT_052", "TRAFFIC_LANES_ON_028A", "DECK_STRUCTURE_TYPE_107",
    "SCOUR_CRITICAL_113", "OWNER_022", "FUNCTIONAL_CLASS_026",
    "DATE_OF_INSPECT_090", "INSPECT_FREQ_MONTHS_091",
    "LAT_016", "LONG_017",
    "DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060", "CULVERT_COND_062",
    "LOWEST_RATING", "BRIDGE_CONDITION",
]

# Each state's rough bounding box, used to validate decoded coordinates.
BOUNDS = {
    "CA": (32.0, 42.5, -125.0, -113.5), "FL": (24.4, 31.1, -87.7, -79.8),
    "MD": (37.8, 39.8, -79.6, -74.9),   "MI": (41.6, 48.3, -90.5, -82.1),
    "NY": (40.4, 45.1, -79.9, -71.7),   "TX": (25.7, 36.6, -106.7, -93.4),
    "WA": (45.5, 49.1, -124.9, -116.9),
}

_DIGITS_ONLY = re.compile(r"^\d+$")


def path(state, year, data_dir=DATA_DIR):
    return data_dir / f"{state}{year}.txt"


def load(state, year, usecols=None, data_dir=DATA_DIR):
    # latin-1 because some older files (for example TX18.txt) are not valid UTF-8
    df = pd.read_csv(path(state, year, data_dir), dtype=str, keep_default_na=False, usecols=usecols,
                     encoding="latin-1")
    df[KEY] = df[KEY].str.strip()
    df["STATE"] = state
    df["YEAR"] = 2000 + int(year)
    return df


def clean_state_year(df):
    """Drop exact duplicate rows and any STRUCTURE_NUMBER_008 that collides across rows."""
    df = df.drop_duplicates()
    dup_keys = df[KEY][df[KEY].duplicated(keep=False)]
    if len(dup_keys):
        df = df[~df[KEY].isin(dup_keys)]
    return df


def decode_lat(x):
    x = x.strip()
    if not _DIGITS_ONLY.match(x):
        return np.nan
    x = int(x)
    dd, mm, ss = x // 1_000_000, (x // 10_000) % 100, (x % 10_000) / 100
    return dd + mm / 60 + ss / 3600


def decode_long(x):
    x = x.strip()
    if not _DIGITS_ONLY.match(x):
        return np.nan
    x = int(x)
    ddd, mm, ss = x // 1_000_000, (x // 10_000) % 100, (x % 10_000) / 100
    return -(ddd + mm / 60 + ss / 3600)


def in_bounds(df):
    """Boolean mask: True where (lat, lon, STATE) falls inside that state's bounding box.
    NaN coordinates evaluate to False here, which is fine -- they're already excluded elsewhere."""
    lo_lat = df["STATE"].map(lambda s: BOUNDS[s][0])
    hi_lat = df["STATE"].map(lambda s: BOUNDS[s][1])
    lo_lon = df["STATE"].map(lambda s: BOUNDS[s][2])
    hi_lon = df["STATE"].map(lambda s: BOUNDS[s][3])
    return df["lat"].between(lo_lat, hi_lat) & df["lon"].between(lo_lon, hi_lon)


def decode_and_validate_coords(df):
    """Add lat/lon columns decoded from LAT_016/LONG_017, masking implausible values to NaN."""
    df["lat"] = df["LAT_016"].apply(decode_lat)
    df["lon"] = df["LONG_017"].apply(decode_long)
    bad_coords = df["lat"].notna() & df["lon"].notna() & ~in_bounds(df)
    df.loc[bad_coords, ["lat", "lon"]] = np.nan
    return df, bad_coords
