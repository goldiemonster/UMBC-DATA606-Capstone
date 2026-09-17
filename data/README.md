# Data

Bridge inventory data from the FHWA [National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm)
(NBI) for the BridgeWatch AI capstone project.

## Contents

- **`{STATE}{YY}.txt`** — 42 raw files: 7 states (CA, FL, MD, MI, NY, TX, WA) × 6 years (2020–2025, encoded
  as `20`–`25`). Comma-delimited, one row per bridge, 123 columns, identical schema across every file. Column
  names embed their official NBI item number from FHWA's *Recording and Coding Guide for the Structure
  Inventory and Appraisal of the Nation's Bridges* (e.g. `YEAR_BUILT_027` = Item 27).
- **`processed/bridge_deterioration_dataset.csv.gz`** — the model-ready dataset built in
  [`../notebooks/eda.ipynb`](../notebooks/eda.ipynb): each row pairs one bridge's record in year *A* (features)
  with whether its condition rating dropped by year *A+1* (`deteriorated_next_period` label). Gzip-compressed
  to stay under GitHub's 100 MB per-file limit; read it with `pd.read_csv(path, compression="gzip")` (pandas
  also infers this automatically from the `.gz` extension).
- **`processed/weather_annual_by_bridge.csv.gz`** — annual climate features per bridge, built by
  [`../scripts/fetch_nclimgrid_weather.py`](../scripts/fetch_nclimgrid_weather.py) from NOAA's
  [nClimGrid-Daily](https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily) gridded product
  (daily tmax/tmin/tavg/prcp, ~5 km resolution, CONUS, public S3 bucket `noaa-nclimgrid-daily-pds`). Covers the
  same **2020–2025** years as the raw NBI inspection files above. Each bridge (`STRUCTURE_NUMBER_008`) is
  matched to its nearest nClimGrid-Daily grid cell using the `lat`/`lon` already decoded in
  `bridge_deterioration_dataset.csv.gz`, and daily values are aggregated per bridge per year into:
  - `tmax_mean_c`, `tmin_mean_c`, `tavg_mean_c` — annual mean daily max/min/average temperature (°C)
  - `prcp_total_mm` — annual total precipitation (mm)
  - `freeze_thaw_days` — count of days where `tmax > 0°C` and `tmin < 0°C` (freeze/thaw cycling, relevant to
    deck and joint deterioration)

  Join onto the bridge dataset via `STRUCTURE_NUMBER_008` + `YEAR`. Raw nClimGrid-Daily netCDF files
  (~4.6 GB total across 72 monthly files) are downloaded and discarded by the script — only this aggregated
  output is stored in the repo.

  **Known gap:** ~1,904 bridges (1.4% of the 140,268 with usable coordinates) have `NaN` temperature columns
  in every year — their nearest grid cell falls outside nClimGrid-Daily's valid land coverage (e.g. immediately
  offshore/at a shoreline). `prcp_total_mm` and `freeze_thaw_days` are unaffected for these rows. Most
  concentrated in FL (575 bridges) and CA (358), consistent with long coastlines.

See [`../docs/proposal.md`](../docs/proposal.md) for the full data dictionary, target/feature definitions, and
data-quality findings.
