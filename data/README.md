# Data

Data for the BridgeWatch AI capstone project, from three public sources: the FHWA
[National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm) (NBI), NOAA's
[nClimGrid-Daily](https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily) weather grids, and FHWA
[Highway Statistics Table SF-12A](https://www.fhwa.dot.gov/policyinformation/statistics/2024/sf12a.cfm) (state
bridge spending).

The bridge and weather data cover 2010 to 2025 for seven states: California, Florida, Maryland, Michigan, New York,
Texas, and Washington. The spending data covers 2020 to 2024.

## Contents

| File | What it is | Rows | Join key |
|---|---|---|---|
| `{STATE}{YY}.txt` (112 files) | Raw bridge inventory records | 2,147,605 in total, one per bridge per year | `STRUCTURE_NUMBER_008` + state |
| `processed/bridge_deterioration_dataset_{YEAR}.csv.gz` (15 files) | Model ready bridge dataset | 1,985,801 in total | `STRUCTURE_NUMBER_008` + `STATE` + `from_year` |
| `weather_annual_by_bridge_{YEAR}.csv` (16 files) | Annual weather per bridge | 2,146,854 in total, one per bridge per year | `STRUCTURE_NUMBER_008` + `STATE` + `YEAR` |
| `sf12a_state_bridge_spending_2020_2024.csv` | Annual bridge spending per state | 35 | `state` + `year` |

### How the files fit together

The raw files feed the model ready dataset, and the other two sources are joined onto it:

- Weather joins on the bridge ID, the state, and the year (`weather.YEAR` matches the dataset's `from_year`). The state
  is part of the key because a structure number is only unique within a state. In 2010, for example, the same number
  belongs to both a Michigan bridge and a Texas bridge.
- Spending joins on the state and the year (`spending.year` matches `from_year`). Every bridge in the same state and
  year gets the same value.

### Raw bridge inventory: `{STATE}{YY}.txt`

112 files: 7 states (CA, FL, MD, MI, NY, TX, WA) by 16 years (2010 to 2025, written as `10` to `25` in the file
name). Comma delimited, one row per bridge. Column names include their official NBI item number from FHWA's
*Recording and Coding Guide for the Structure Inventory and Appraisal of the Nation's Bridges* (for example,
`YEAR_BUILT_027` is Item 27).

The files do not all have the same columns, because FHWA changed its format over this period:

| Years | Columns |
|---|---|
| 2010 to 2015 | 134 |
| 2016 | 135 |
| 2017 and 2018 | 137 |
| 2019 to 2025 | 123 |

Every column the project uses is present in all 112 files, with two exceptions that are rebuilt. FHWA only publishes
`LOWEST_RATING` and `BRIDGE_CONDITION` (summaries of the four component condition ratings) from 2019 on. For
2010 to 2018, `eda.ipynb` rebuilds them from the component ratings (`DECK_COND_058`, `SUPERSTRUCTURE_COND_059`,
`SUBSTRUCTURE_COND_060`, `CULVERT_COND_062`). The same rule reproduces FHWA's published values exactly on the
959,519 records from 2019 to 2025.

Two practical notes for reading the files with pandas. Read them with `encoding="latin-1"`, since at least one file
(`TX18.txt`) is not valid UTF-8. And read the code columns as text (`dtype=str`), so a code like `5` is never turned
into `5.0`.

### Model ready bridge dataset: `processed/bridge_deterioration_dataset_{YEAR}.csv.gz`

Built in [`../notebooks/eda.ipynb`](../notebooks/eda.ipynb). Each row pairs one bridge's record in year *A* (the
features) with whether its condition rating dropped by year *A+1* (the `deteriorated_next_period` label). There are
15 files, one for each feature year (`from_year` 2010 through 2024), so together they hold the transitions from
2010 to 11 through 2024 to 25. Each file is 4 to 5 MB, which keeps it under GitHub's 25 MB limit for uploads through
the web page. The rows per file grow from about 117,000 to about 138,000. Stack them with `pd.concat` to rebuild the
full table. pandas reads each one directly: `pd.read_csv(path, compression="gzip")`.

The table has 39 columns: the bridge's features in year *A*, its rating in year *B* (`LOWEST_RATING_next`,
`BRIDGE_CONDITION_next`), the years (`from_year`, `to_year`), the label, and a few derived columns (`age`, `lat`,
`lon`, `months_since_inspection`). About 5.3% of rows have the label set to 1.

Some things about how it was built:

- Latitude and longitude are decoded from NBI's packed format and checked against each state's bounding box. A
  coordinate that is missing or implausible in one year is filled in from the same bridge's other years. After that,
  only 595 of the 1,985,801 rows have no usable coordinate.
- 46 rows were dropped from 8 raw files because the same structure number appeared twice with different records, and
  there is no way to tell which one is right.
- Michigan's 2010 file numbers its bridges differently from every later year, so only 80 of its 10,928 bridges match
  the 2011 file. The Michigan 2010 to 11 transitions are almost all missing, about 11,000 rows.
- California's ratings swing sharply around 2016 to 2020. About a quarter of its ratings went up in 2016, and then
  23% and 25% of its bridges dropped in 2017 and 2018. Those rows are kept here, but the training notebook leaves out
  California transitions ending 2016 to 2020, because training on them made the model worse.

### Weather: `weather_annual_by_bridge_2010.csv` through `weather_annual_by_bridge_2025.csv`

Annual climate summaries for each bridge, built by
[`../scripts/fetch_nclimgrid_weather.py`](../scripts/fetch_nclimgrid_weather.py) from NOAA's nClimGrid-Daily
product (daily temperature and precipitation on a grid of about 5 km, for the contiguous U.S.). Each bridge is
matched to its nearest grid cell, using the latest year in which the bridge has a valid coordinate, and daily values
are summarized for each year:

- `tmax_mean_c`, `tmin_mean_c`, `tavg_mean_c`: annual mean of daily maximum, minimum, and average temperature (°C)
- `prcp_total_mm`: annual total precipitation (mm)
- `freeze_thaw_days`: number of days where `tmax > 0°C` and `tmin < 0°C` (freeze and thaw cycling, which is hard on
  bridge decks and joints)

Each file has one row per bridge that appears in the inventory that year, so the row count grows from 129,039 in
2010 to 139,026 in 2025. Bridges with no usable coordinate in any year have no weather row. The 16 files have
identical columns (`STRUCTURE_NUMBER_008`, `STATE`, `YEAR`, and the five values above), and each is about 9 MB so it
is easy to upload and open. Stack them with `pd.concat` to rebuild the full table. The raw weather grids (roughly
12 GB across 192 monthly files) are downloaded and then deleted by the script, and are not stored here.

**Known gap:** about 1,700 to 1,900 bridges each year (1.3% to 1.4%) have all five weather values blank, because
their nearest grid cell falls just offshore or outside the grid's land coverage. The same bridges are blank in every
year. In 2025 the gap is largest in Florida (565 bridges) and California (351), which fits their long coastlines.
These values are left blank rather than filled with zeros, so a precipitation of 0 always means a measured zero.

### State bridge spending: `sf12a_state_bridge_spending_2020_2024.csv`

What each state highway agency spent on bridge work, in thousands of dollars, cleaned from FHWA's Table SF-12A for
2020 through 2024 (2025 has not been published yet, and 2010 to 2019 have not been collected). Because of that, the
spending experiments in the training notebook use only the 2020 and later transitions.

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
a state has. The notebooks use spending per inventoried bridge instead of the raw total.

See [`../docs/proposal.md`](../docs/proposal.md) for the full data dictionary, target and feature definitions, and
data quality findings.
