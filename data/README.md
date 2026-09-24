# Data

Data for the BridgeWatch AI capstone project, from three public sources: the FHWA
[National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm) (NBI), NOAA's
[nClimGrid-Daily](https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily) weather grids, and FHWA
[Highway Statistics Table SF-12A](https://www.fhwa.dot.gov/policyinformation/statistics/2024/sf12a.cfm) (state
bridge spending).

## Contents

| File | What it is | Rows | Join key |
|---|---|---|---|
| `{STATE}{YY}.txt` (42 files) | Raw bridge inventory records | One per bridge per year | `STRUCTURE_NUMBER_008` |
| `processed/bridge_deterioration_dataset.csv.gz` | Model ready bridge dataset | 681,841 | `STRUCTURE_NUMBER_008` + `from_year` |
| `weather_annual_by_bridge_{YEAR}.csv` (6 files) | Annual weather per bridge | 140,268 per file | `STRUCTURE_NUMBER_008` + `YEAR` |
| `sf12a_state_bridge_spending_2020_2024.csv` | Annual bridge spending per state | 35 | `state` + `year` |

### Raw bridge inventory: `{STATE}{YY}.txt`

42 files: 7 states (CA, FL, MD, MI, NY, TX, WA) by 6 years (2020 to 2025, written as `20` to `25` in the file
name). Comma delimited, one row per bridge, 123 columns, with the same columns in every file. Column names include
their official NBI item number from FHWA's *Recording and Coding Guide for the Structure Inventory and Appraisal of
the Nation's Bridges* (for example, `YEAR_BUILT_027` is Item 27).

### Model ready bridge dataset: `processed/bridge_deterioration_dataset.csv.gz`

Built in [`../notebooks/eda.ipynb`](../notebooks/eda.ipynb). Each row pairs one bridge's record in year *A* (the
features) with whether its condition rating dropped by year *A+1* (the `deteriorated_next_period` label). It is
gzip compressed to stay under GitHub's 100 MB per file limit. pandas reads it directly:
`pd.read_csv(path, compression="gzip")`.

### Weather: `weather_annual_by_bridge_2020.csv` through `weather_annual_by_bridge_2025.csv`

Annual climate summaries for each bridge, built by
[`../scripts/fetch_nclimgrid_weather.py`](../scripts/fetch_nclimgrid_weather.py) from NOAA's nClimGrid-Daily
product (daily temperature and precipitation on a grid of about 5 km, for the contiguous U.S.). Each bridge is
matched to its nearest grid cell using the `lat` and `lon` decoded in the bridge dataset, and daily values are
summarized for each year:

- `tmax_mean_c`, `tmin_mean_c`, `tavg_mean_c`: annual mean of daily maximum, minimum, and average temperature (°C)
- `prcp_total_mm`: annual total precipitation (mm)
- `freeze_thaw_days`: number of days where `tmax > 0°C` and `tmin < 0°C` (freeze and thaw cycling, which is hard on
  bridge decks and joints)

The script writes one compressed file (`processed/weather_annual_by_bridge.csv.gz`, about 78 MB uncompressed).
It is stored here split into one plain csv per year, about 13 MB each, so each file is easy to upload and open.
The six files have identical columns; stack them with `pd.concat` to rebuild the full table. The raw weather grids
(about 4.6 GB across 72 monthly files) are downloaded and then deleted by the script, and are not stored here.

**Known gap:** about 1,904 bridges (1.4% of the 140,268 with usable coordinates) have missing temperature values
in every year, because their nearest grid cell falls just offshore or outside the grid's land coverage.
`prcp_total_mm` and `freeze_thaw_days` are not affected. The gap is largest in Florida (575 bridges) and California
(358), which fits their long coastlines.

### State bridge spending: `sf12a_state_bridge_spending_2020_2024.csv`

What each state highway agency spent on bridge work, in thousands of dollars, cleaned from FHWA's Table SF-12A for
2020 through 2024 (2025 has not been published yet).

| Column | Meaning |
|---|---|
| `state` | Two letter state code |
| `year` | Spending year, 2020 to 2024 |
| `bridge_replacement_thousands` | Replacing existing bridges |
| `major_bridge_rehab_thousands` | Major bridge rehabilitation |
| `minor_bridge_rehab_thousands` | Minor bridge rehabilitation |
| `bridge_repair_total_thousands` | Replacement plus major and minor rehabilitation |
| `new_bridge_thousands` | New bridge construction, kept out of the repair total |

How it was built: each year's workbook has 14 sheets, one per road system and area (rural and urban interstates,
other freeways and expressways, principal arterials, minor arterials, major and minor collectors, and local roads).
The four bridge columns were summed across all 14 sheets for each state and year, and only the seven project
states were kept. Each parsed row was checked against the row total FHWA reports, and all 490 state, year, and
sheet combinations matched. Summing all 14 sheets matters: the rural interstate sheet on its own covers only about
6% of bridge repair spending in these states.

Keep in mind that these are raw dollars, not adjusted for inflation, and that they mostly reflect how many bridges
a state has. The model uses spending per inventoried bridge instead of the raw total.

See [`../docs/proposal.md`](../docs/proposal.md) for the full data dictionary, target and feature definitions, and
data quality findings.
