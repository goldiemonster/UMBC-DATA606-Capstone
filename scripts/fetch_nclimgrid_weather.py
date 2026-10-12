"""Build per-bridge, per-year weather features from NOAA nClimGrid-Daily.

Pulls the NOAA nClimGrid-Daily gridded climate product (daily tmax/tmin/tavg/prcp, ~5 km
resolution, contiguous U.S., public S3 bucket `noaa-nclimgrid-daily-pds`, no login needed),
matches every bridge to its nearest grid cell, and summarizes the daily values into one row
per bridge per year:

- tmax_mean_c, tmin_mean_c, tavg_mean_c: annual mean temperatures (Celsius)
- prcp_total_mm: annual total precipitation (mm)
- freeze_thaw_days: days where tmax > 0C and tmin < 0C

A bridge's location comes from the raw inspection files in --data-dir ({STATE}{YY}.txt). The
coordinates are decoded and validated the same way as in eda.ipynb, and each bridge uses its
coordinates from the most recent year (of all years found in --data-dir) in which they are
usable, so a bad coordinate in one year is filled in from another year. One output row is
written for every bridge that appears in that year's inspection files.

If a bridge's grid cell has no data at all (for example a cell just offshore), all five weather
values are left empty. They are NOT filled with zeros.

Each monthly netCDF file (~64 MB) is downloaded, used, and deleted. Output is one csv per year,
named weather_annual_by_bridge_{YEAR}.csv.

Usage:
    python scripts/fetch_nclimgrid_weather.py --years 2010-2019
    python scripts/fetch_nclimgrid_weather.py --years 2010-2025 --data-dir data --out-dir data
"""

import argparse
import concurrent.futures as cf
import http.client
import re
import shutil
import time
import urllib.request
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

import nbi_utils as nbi

BASE_URL = "https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/access/grids/{year}/ncdd-{year}{month:02d}-grd-scaled.nc"
RAW_NAME = re.compile(r"^(CA|FL|MD|MI|NY|TX|WA)(\d{2})\.txt$")
ACC_KEYS = ["tmax_sum", "tmin_sum", "tavg_sum", "prcp_sum", "freeze_thaw", "n_temp_days", "n_prcp_days"]


def parse_years(text):
    years = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def read_bridge_file(path, state):
    """Bridge IDs and decoded, validated lat/lon from one raw inspection file."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1",
                     usecols=[nbi.KEY, "LAT_016", "LONG_017"])
    df[nbi.KEY] = df[nbi.KEY].str.strip()
    df["STATE"] = state
    df = nbi.clean_state_year(df)
    df, _ = nbi.decode_and_validate_coords(df)
    return df[["STATE", nbi.KEY, "lat", "lon"]]


def load_bridges(data_dir, target_years):
    """Per-year bridge lists, plus one best-known location per bridge."""
    found = {}
    for p in sorted(Path(data_dir).iterdir()):
        m = RAW_NAME.match(p.name)
        if m:
            found[(m.group(1), 2000 + int(m.group(2)))] = p
    all_years = sorted({y for _, y in found})
    for year in target_years:
        missing = [s for s in nbi.STATES if (s, year) not in found]
        if missing:
            raise SystemExit(f"Missing raw bridge files for {year}: {missing}")

    frames, presence = [], {}
    for year in all_years:
        parts = [read_bridge_file(found[(s, year)], s) for s in nbi.STATES if (s, year) in found]
        df = pd.concat(parts, ignore_index=True)
        frames.append(df.assign(year=year))
        if year in target_years:
            presence[year] = df[["STATE", nbi.KEY]]
        print(f"Read bridge records for {year}: {len(df):,} bridges, {int(df['lat'].isna().sum()):,} without usable coordinates")

    allrows = pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon"])
    locations = (allrows.sort_values("year")
                 .drop_duplicates(["STATE", nbi.KEY], keep="last")[["STATE", nbi.KEY, "lat", "lon"]])
    return presence, locations


def nearest_grid_index(values, grid):
    """Index into the sorted grid array nearest each value."""
    idx = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    idx -= (values - grid[idx - 1]) < (grid[idx] - values)
    return idx


def download_month(year, month, work_dir):
    """Download one monthly file. Transfers are sometimes cut short, so keep resuming where it stopped."""
    url = BASE_URL.format(year=year, month=month)
    dest = Path(work_dir) / f"ncdd-{year}{month:02d}.nc"
    dest.unlink(missing_ok=True)
    total, have = None, 0
    for attempt in range(1, 31):
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                if resp.status == 206:
                    total = int(resp.headers["Content-Range"].split("/")[1])
                    mode = "ab"
                else:  # the server sent the whole file (again)
                    total = int(resp.headers["Content-Length"])
                    mode = "wb"
                with open(dest, mode) as out:
                    shutil.copyfileobj(resp, out)
        except (OSError, http.client.HTTPException) as exc:
            print(f"  download of {year}-{month:02d} interrupted (attempt {attempt}): {exc}")
        have = dest.stat().st_size if dest.exists() else 0
        if total is not None and have == total:
            return dest
        time.sleep(min(2 ** attempt, 20))
    raise SystemExit(f"Could not download {url}")


def new_accumulators(n_cells):
    return {k: np.zeros(n_cells) for k in ACC_KEYS}


def add_month(path, acc, lat_idx, lon_idx, grid_lat, grid_lon):
    with nc.Dataset(path) as ds:
        if not (np.allclose(ds.variables["lat"][:], grid_lat) and np.allclose(ds.variables["lon"][:], grid_lon)):
            raise SystemExit(f"{path.name} is on a different grid than the first file")
        # netCDF4 treats two index arrays as an outer product, so subset with numpy instead.
        tmax = ds.variables["tmax"][:][:, lat_idx, lon_idx]
        tmin = ds.variables["tmin"][:][:, lat_idx, lon_idx]
        tavg = ds.variables["tavg"][:][:, lat_idx, lon_idx]
        prcp = ds.variables["prcp"][:][:, lat_idx, lon_idx]
    valid_t = ~(np.ma.getmaskarray(tmax) | np.ma.getmaskarray(tmin) | np.ma.getmaskarray(tavg))
    valid_p = ~np.ma.getmaskarray(prcp)
    acc["tmax_sum"] += np.where(valid_t, tmax, 0).sum(axis=0)
    acc["tmin_sum"] += np.where(valid_t, tmin, 0).sum(axis=0)
    acc["tavg_sum"] += np.where(valid_t, tavg, 0).sum(axis=0)
    acc["prcp_sum"] += np.where(valid_p, prcp, 0).sum(axis=0)
    acc["freeze_thaw"] += ((np.ma.filled(tmax, 0) > 0) & (np.ma.filled(tmin, 0) < 0) & valid_t).sum(axis=0)
    acc["n_temp_days"] += valid_t.sum(axis=0)
    acc["n_prcp_days"] += valid_p.sum(axis=0)


def annual_stats(acc):
    """Turn running totals into annual values; cells with no data stay empty (NaN)."""
    nt = np.where(acc["n_temp_days"] == 0, np.nan, acc["n_temp_days"])
    npr = np.where(acc["n_prcp_days"] == 0, np.nan, acc["n_prcp_days"])
    return pd.DataFrame({
        "tmax_mean_c": acc["tmax_sum"] / nt,
        "tmin_mean_c": acc["tmin_sum"] / nt,
        "tavg_mean_c": acc["tavg_sum"] / nt,
        "prcp_total_mm": np.where(np.isnan(npr), np.nan, acc["prcp_sum"]),
        "freeze_thaw_days": np.where(np.isnan(nt), np.nan, acc["freeze_thaw"]),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2010-2025", help="years to build, e.g. 2010-2019 or 2012,2016")
    ap.add_argument("--data-dir", default=str(nbi.DATA_DIR), help="folder with the raw {STATE}{YY}.txt files")
    ap.add_argument("--out-dir", default=str(nbi.DATA_DIR), help="where the yearly csv files are written")
    ap.add_argument("--work-dir", default=None, help="scratch folder for downloads and progress files")
    args = ap.parse_args()

    years = parse_years(args.years)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else out_dir / ".nclimgrid_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    presence, locations = load_bridges(args.data_dir, years)
    needed = pd.concat(presence.values()).drop_duplicates()
    locations = needed.merge(locations, on=["STATE", nbi.KEY], how="left")
    n_no_loc = int(locations["lat"].isna().sum())
    print(f"\n{len(locations):,} distinct bridges in {years[0]}-{years[-1]}; "
          f"{n_no_loc:,} have no usable coordinates in any year and are left out")
    locations = locations.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    first = work_dir / "grid_reference.nc"
    if not first.exists():
        shutil.move(download_month(years[0], 1, work_dir), first)
    with nc.Dataset(first) as ds:
        grid_lat = ds.variables["lat"][:].astype(np.float64)
        grid_lon = ds.variables["lon"][:].astype(np.float64)

    locations["lat_idx"] = nearest_grid_index(locations["lat"].to_numpy(), grid_lat)
    locations["lon_idx"] = nearest_grid_index(locations["lon"].to_numpy(), grid_lon)
    locations["cell_key"] = locations["lat_idx"].astype(np.int64) * 100000 + locations["lon_idx"].astype(np.int64)
    cells = locations.drop_duplicates("cell_key")[["cell_key", "lat_idx", "lon_idx"]].reset_index(drop=True)
    cell_key, lat_idx, lon_idx = cells["cell_key"].to_numpy(), cells["lat_idx"].to_numpy(), cells["lon_idx"].to_numpy()
    print(f"{len(cells):,} distinct grid cells to read\n")

    # Monthly files are the slow part, so each year's running totals are saved when the year is done.
    done = {}
    for year in years:
        ckpt = work_dir / f"totals_{year}.npz"
        if ckpt.exists():
            saved = np.load(ckpt)
            if np.array_equal(saved["cell_key"], cell_key):
                done[year] = {k: saved[k] for k in ACC_KEYS}
                print(f"{year}: using saved progress")
    todo = [(y, m) for y in years if y not in done for m in range(1, 13)]

    acc = None
    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        nxt = pool.submit(download_month, *todo[0], work_dir) if todo else None
        for i, (year, month) in enumerate(todo):
            path = nxt.result()
            nxt = pool.submit(download_month, *todo[i + 1], work_dir) if i + 1 < len(todo) else None
            if month == 1:
                acc = new_accumulators(len(cells))
            add_month(path, acc, lat_idx, lon_idx, grid_lat, grid_lon)
            path.unlink()
            print(f"{year}-{month:02d} done")
            if month == 12:
                np.savez(work_dir / f"totals_{year}.npz", cell_key=cell_key, **acc)
                done[year] = acc

    for year in years:
        stats = annual_stats(done[year])
        stats.insert(0, "cell_key", cell_key)
        expected = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
        partial = int(((done[year]["n_temp_days"] > 0) & (done[year]["n_temp_days"] < expected)).sum())
        empty = int((done[year]["n_temp_days"] == 0).sum())
        rows = presence[year].merge(locations[["STATE", nbi.KEY, "cell_key"]], on=["STATE", nbi.KEY], how="inner")
        out = rows.merge(stats, on="cell_key", how="left").drop(columns="cell_key")
        out.insert(2, "YEAR", year)
        out = out[[nbi.KEY, "STATE", "YEAR", "tmax_mean_c", "tmin_mean_c", "tavg_mean_c", "prcp_total_mm", "freeze_thaw_days"]]
        out = out.sort_values(nbi.KEY).reset_index(drop=True).round(6)
        path = out_dir / f"weather_annual_by_bridge_{year}.csv"
        out.to_csv(path, index=False)
        print(f"{year}: wrote {len(out):,} rows to {path.name} ({path.stat().st_size / 1e6:.1f} MB); "
              f"{empty} cells with no data, {partial} cells with partial-year data")

    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
