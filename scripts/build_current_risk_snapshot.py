"""Build the current (2025) per-bridge feature snapshot the Streamlit app scores.

The model is trained on transitions that start in 2010 to 2024, and 2025 is only ever used to check whether a
bridge deteriorated. To show forward-looking risk in the app, we need each bridge's most recent snapshot: the 2025
NBI features plus the 2025 weather, with no label. It uses the same columns and cleaning as the training feature
set (see notebooks/model_training_step_9.ipynb), so the app can feed rows straight into the trained model without
reparsing raw NBI files or weather netCDF grids at runtime.

Usage:
    python scripts/build_current_risk_snapshot.py
"""

import gzip

import numpy as np
import pandas as pd

import nbi_utils as nbi

SNAPSHOT_WEATHER_PATH = nbi.DATA_DIR / "weather_annual_by_bridge_2025.csv"
OUTPUT_PATH = nbi.DATA_DIR / "processed" / "bridge_current_risk_input.csv.gz"
SNAPSHOT_YEAR = 2025


def load_snapshot() -> pd.DataFrame:
    year_code = str(SNAPSHOT_YEAR)[-2:]
    frames = [nbi.clean_state_year(nbi.load(state, year_code, usecols=nbi.FEATURE_COLS))
              for state in nbi.STATES]
    df = pd.concat(frames, ignore_index=True)
    df, bad_coords = nbi.decode_and_validate_coords(df)
    print(f"Coordinates masked as implausible: {bad_coords.sum()} / {len(df)}")

    df["age"] = SNAPSHOT_YEAR - pd.to_numeric(df["YEAR_BUILT_027"], errors="coerce")
    df["ADT_029"] = pd.to_numeric(df["ADT_029"], errors="coerce")
    df["PERCENT_ADT_TRUCK_109"] = pd.to_numeric(df["PERCENT_ADT_TRUCK_109"], errors="coerce")
    df["MAIN_UNIT_SPANS_045"] = pd.to_numeric(df["MAIN_UNIT_SPANS_045"], errors="coerce")
    df["STRUCTURE_LEN_MT_049"] = pd.to_numeric(df["STRUCTURE_LEN_MT_049"], errors="coerce")
    df["DECK_WIDTH_MT_052"] = pd.to_numeric(df["DECK_WIDTH_MT_052"], errors="coerce")
    df["TRAFFIC_LANES_ON_028A"] = pd.to_numeric(df["TRAFFIC_LANES_ON_028A"], errors="coerce")
    df["INSPECT_FREQ_MONTHS_091"] = pd.to_numeric(df["INSPECT_FREQ_MONTHS_091"], errors="coerce")
    df["LOWEST_RATING"] = pd.to_numeric(df["LOWEST_RATING"], errors="coerce")
    return df


def fill_missing_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing 2025 coordinate from the same bridge's earlier years.

    A bridge does not move, and the training data was cleaned the same way (see eda.ipynb, Section 6). Only gaps
    are filled; a valid 2025 coordinate is never overwritten.
    """
    df = df.copy()
    df.loc[df["lat"].isna() | df["lon"].isna(), ["lat", "lon"]] = np.nan
    before = int(df["lat"].isna().sum())
    for year in reversed(nbi.YEARS[:-1]):  # 2024 back to 2010
        needs = df["lat"].isna()
        if not needs.any():
            break
        for state in nbi.STATES:
            wanted = set(df.loc[needs & (df["STATE"] == state), nbi.KEY])
            if not wanted:
                continue
            old = nbi.load(state, year, usecols=[nbi.KEY, "LAT_016", "LONG_017"])
            old = old[old[nbi.KEY].isin(wanted)]
            if old.empty:
                continue
            old, _ = nbi.decode_and_validate_coords(old)
            old = old.dropna(subset=["lat", "lon"]).drop_duplicates(nbi.KEY, keep=False).set_index(nbi.KEY)
            rows = df.index[needs & (df["STATE"] == state) & df[nbi.KEY].isin(old.index)]
            df.loc[rows, ["lat", "lon"]] = old.loc[df.loc[rows, nbi.KEY], ["lat", "lon"]].to_numpy()
    print(f"Bridges with no usable coordinate: {before} in {SNAPSHOT_YEAR}, "
          f"{int(df['lat'].isna().sum())} after filling from earlier years")
    return df


def main() -> None:
    df = load_snapshot()
    print(f"Loaded {len(df)} bridges in the {SNAPSHOT_YEAR} snapshot")
    df = fill_missing_coordinates(df)

    weather = pd.read_csv(SNAPSHOT_WEATHER_PATH)
    weather_year = weather.loc[weather["YEAR"] == SNAPSHOT_YEAR].drop(columns="YEAR")
    # A structure number is only unique within a state, so the state is part of the join key
    df = df.merge(weather_year, on=[nbi.KEY, "STATE"], how="left")
    match_rate = df["tavg_mean_c"].notna().mean() * 100
    print(f"Weather match rate: {match_rate:.1f}%")

    keep_cols = [
        nbi.KEY, "STATE",
        "age", "ADT_029", "PERCENT_ADT_TRUCK_109", "STRUCTURE_KIND_043A", "STRUCTURE_TYPE_043B",
        "MAIN_UNIT_SPANS_045", "STRUCTURE_LEN_MT_049", "DECK_WIDTH_MT_052", "TRAFFIC_LANES_ON_028A",
        "DECK_STRUCTURE_TYPE_107", "SCOUR_CRITICAL_113", "OWNER_022", "INSPECT_FREQ_MONTHS_091",
        "DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060", "CULVERT_COND_062",
        "LOWEST_RATING", "BRIDGE_CONDITION", "lat", "lon",
        "tmax_mean_c", "tmin_mean_c", "tavg_mean_c", "prcp_total_mm", "freeze_thaw_days",
    ]
    df = df[keep_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", newline="") as f:
        df.to_csv(f, index=False)

    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
