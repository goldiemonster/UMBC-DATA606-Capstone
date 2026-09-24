# BridgeWatch AI: Predicting Bridge Condition Deterioration from Federal Inspection Records

## 1. Title and Author

- **Project Title:** BridgeWatch AI: Predicting Bridge Condition Deterioration from Federal Inspection Records
- Prepared for the UMBC Data Science Master's Degree Capstone by Dr. Chaojie (Jay) Wang
- **Author:** Edmund L. Goldsberry
- **GitHub repository:** https://github.com/goldiemonster/UMBC-DATA606-Capstone


## 2. Background

### What is it about?

Every U.S. highway bridge is inspected on a regular cycle (generally every 12 to 24 months) under the National
Bridge Inspection Standards, and the results are reported to the Federal Highway Administration (FHWA) as part of
the **National Bridge Inventory (NBI)**. The NBI is a structured record, for every bridge in the country, of its
physical characteristics (age, material, span, traffic volume) and its condition ratings (deck, superstructure,
substructure, and an overall condition category of Good, Fair, or Poor).

BridgeWatch AI uses six consecutive years of NBI data (2020 to 2025) for seven states to build a model that
predicts whether a bridge's condition rating will **decline by its next inspection cycle**, before that decline
shows up in an inspection report. The bridge records are combined with annual weather data for each bridge's
location and with each state's annual spending on bridge repair. The end product is a Streamlit application that
lets a user look up a bridge (or browse a state map of bridges color coded by predicted risk) and see a
deterioration risk score along with the factors driving it.

### Why does it matter?

The 2021 Infrastructure Investment and Jobs Act highlighted how much of the U.S. bridge inventory is aging past
its intended design life, and bridge maintenance budgets are finite: state DOTs cannot re-inspect or rehabilitate
every structure every year. A model that flags which bridges are statistically likely to deteriorate soon, rather
than waiting for the next scheduled inspection to find out, gives inspection planners a data driven way to direct
limited inspection and maintenance resources toward the structures most likely to need them, instead of relying
only on fixed inspection intervals and age based rules of thumb.

### Research questions

1. Can machine learning predict whether a bridge's condition rating will decline by its next inspection cycle,
   using only information available at the time of the current inspection (structural characteristics, traffic
   loading, age, and current condition)?
2. Which factors are the strongest predictors of near term deterioration: age, traffic volume and truck loading,
   construction material, scour vulnerability, or time since last inspection?
3. Does deterioration risk vary meaningfully by state or region, suggesting climate effects beyond what is
   captured in the bridge's own attributes?
4. Is a state's spending on bridge repair related to how quickly its bridges deteriorate, and does including it
   improve predictions?

## 3. Data

### Data sources

The project combines three public data sources.

**1. [FHWA National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm).** The official, publicly
published ASCII bridge inventory files FHWA releases annually for every U.S. state, territory, and federal agency,
submitted under the National Bridge Inspection Standards (23 CFR 650 Subpart C). Field definitions follow FHWA's
*Recording and Coding Guide for the Structure Inventory and Appraisal of the Nation's Bridges* (each column name
embeds its official NBI item number, for example `YEAR_BUILT_027` is Item 27).

This project uses **7 states** (California, Florida, Maryland, Michigan, New York, Texas, Washington), chosen for
geographic and climate diversity, each with **6 annual snapshots (2020 to 2025)**, for 42 raw files in total.

**2. [NOAA nClimGrid-Daily](https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily).** NCEI's daily
gridded climate product (daily maximum, minimum, and average temperature and precipitation, at about 5 km
resolution, for the contiguous U.S., 1951 to present), distributed as monthly netCDF files in the public S3 bucket
`noaa-nclimgrid-daily-pds`. Used for research question 3. See the weather section below.

**3. [FHWA Highway Statistics, Table SF-12A](https://www.fhwa.dot.gov/policyinformation/statistics/2024/sf12a.cfm).**
State highway agency capital outlay by improvement type, in thousands of dollars, published annually. Used for
research question 4. See the spending section below.

### Data size

- Raw bridge files: 42 `.txt` files (comma delimited), **about 326 MB** combined, stored in `data/`.
- Model ready bridge dataset: `data/processed/bridge_deterioration_dataset.csv.gz` (**about 27 MB** compressed,
  about 118 MB uncompressed; compressed to stay under GitHub's 100 MB per file limit).
- Weather features: six files, `data/weather_annual_by_bridge_2020.csv` through `_2025.csv`, **about 13 MB**
  each (split by year so each file stays small enough to upload easily).
- State bridge spending: `data/sf12a_state_bridge_spending_2020_2024.csv`, **35 rows** (7 states by 5 years).

### Data shape

- Each raw bridge file: **123 columns**, one row per bridge. Row counts range from 5,430 (Maryland, 2020) to
  56,951 (Texas, 2025).
- Combined 2020 to 2025 raw data: **824,312 bridge year records** across all 42 files.
- The **model ready dataset**, built by pairing each bridge's consecutive annual records (2020 to 21, 21 to 22,
  and so on through 24 to 25) within each state: **681,841 rows by 39 columns**.
- Weather: **841,608 rows** (140,268 bridges by 6 years) by 8 columns.
- Spending: **35 rows** by 7 columns.

### Time period

2020 to 2025 (6 calendar years) for bridges and weather, yielding 5 year over year transitions per state that the
model learns from. Spending covers 2020 to 2024; 2025 has not been published yet. Because each transition's
features come from its starting year (2020 through 2024), every row in the model dataset has matching spending
data.

### What does each row represent?

- In a **raw bridge** file: one row is one bridge structure, as inventoried by that state in that year.
- In the **model ready** dataset: one row is one bridge's **transition** from one annual inspection to the next
  (for example, "Bridge X's 2022 record: did its condition rating drop by 2023?"). The same physical bridge
  contributes up to 5 such rows, one per consecutive year pair it appears in.
- In the **weather** data: one row is one bridge in one calendar year.
- In the **spending** data: one row is one state in one year.

### Weather data

Pulled for the same 2020 to 2025 period and 7 state footprint as the bridge data. Each bridge is matched to its
nearest nClimGrid-Daily grid cell using its decoded `lat` and `lon`, and daily values are summarized into one row
per bridge per year by [`scripts/fetch_nclimgrid_weather.py`](../scripts/fetch_nclimgrid_weather.py). The raw
netCDF grids (about 4.6 GB across 72 monthly files) are downloaded and discarded by the script; only the small
summarized output is stored in the repo.

| Column | Type | Definition | Values |
|---|---|---|---|
| `tmax_mean_c` | float | Annual mean of daily maximum temperature | °C |
| `tmin_mean_c` | float | Annual mean of daily minimum temperature | °C |
| `tavg_mean_c` | float | Annual mean of daily average temperature | °C |
| `prcp_total_mm` | float | Annual total precipitation | mm, 0 or more |
| `freeze_thaw_days` | int | Count of days where `tmax > 0°C` and `tmin < 0°C` | 0 to 365 |

Joins onto the bridge dataset on `STRUCTURE_NUMBER_008` plus year. About 1.4% of bridges have missing
temperature values because their nearest grid cell falls just offshore or outside the grid's land coverage.

### State bridge spending data

Each year's SF-12A workbook has 14 sheets, one per road system and area (rural and urban interstates, other
freeways and expressways, principal arterials, minor arterials, major and minor collectors, and local roads). Four
columns on every sheet cover bridge work. The cleaned file sums each column across all 14 sheets for each state
and year. Every parsed row was checked against the row total FHWA reports, and all 490 state, year, and sheet
combinations matched.

| Column | Type | Definition | Values |
|---|---|---|---|
| `state` | categorical | Two letter state code | CA, FL, MD, MI, NY, TX, WA |
| `year` | int | Spending year | 2020 to 2024 |
| `bridge_replacement_thousands` | float | Spending on replacing existing bridges | Thousands of dollars |
| `major_bridge_rehab_thousands` | float | Spending on major bridge rehabilitation | Thousands of dollars |
| `minor_bridge_rehab_thousands` | float | Spending on minor bridge rehabilitation | Thousands of dollars |
| `bridge_repair_total_thousands` | float | Replacement plus major and minor rehabilitation | Thousands of dollars |
| `new_bridge_thousands` | float | Spending on new bridge construction, kept separate from the repair total | Thousands of dollars |

Joins onto the bridge dataset on `STATE` plus `from_year`, so every bridge in the same state and year receives
the same values.

Two cautions shape how this data is used. First, raw dollars mostly reflect state size (Texas has about ten times
as many bridges as Maryland), so the model uses spending per inventoried bridge rather than the raw total.
Second, states tend to spend more where bridges are already in poor condition, so a link between spending and
deterioration in this data should not be read as spending causing it.

### Data dictionary (bridge inventory)

Full field definitions are in FHWA's Recording and Coding Guide (linked above). The bridge columns used in this
project are:

| Column | Type | Definition | Values |
|---|---|---|---|
| `STRUCTURE_NUMBER_008` | string | Unique bridge identifier (join key across years) | Free form, state assigned |
| `STATE` (derived) | categorical | Two letter state code | CA, FL, MD, MI, NY, TX, WA |
| `YEAR_BUILT_027` | int | Year the bridge was originally built | For example 1937 to 2025 |
| `age` (derived) | int | Inspection year minus `YEAR_BUILT_027` | 0 to 225 |
| `ADT_029` | int | Average Daily Traffic | 0 to 810,110 |
| `PERCENT_ADT_TRUCK_109` | float | Percent of ADT that is truck traffic | 0 to 100 |
| `STRUCTURE_KIND_043A` | categorical | Primary material and design (NBI Item 43A) | 0=Other, 1/2=Concrete (simple/continuous), 3/4=Steel (simple/continuous), 5/6=Prestressed concrete (simple/continuous), 7=Wood/timber, 8=Masonry, 9=Aluminum/iron |
| `STRUCTURE_TYPE_043B` | categorical | Structure type and design (NBI Item 43B) | Coded per Recording and Coding Guide |
| `MAIN_UNIT_SPANS_045` | int | Number of spans in main unit | 1 or more |
| `STRUCTURE_LEN_MT_049` | float | Total structure length (meters) | Greater than 0 |
| `DECK_WIDTH_MT_052` | float | Deck width (meters) | Greater than 0 |
| `TRAFFIC_LANES_ON_028A` | int | Number of traffic lanes on the structure | 0 or more |
| `SCOUR_CRITICAL_113` | categorical | Scour vulnerability rating | N, U, 0 to 9 |
| `OWNER_022` | categorical | Owning agency type | Coded per Recording and Coding Guide |
| `INSPECT_FREQ_MONTHS_091` | int | Months between required inspections | Typically 12 to 48 |
| `LAT_016` / `LONG_017` (decoded to `lat`/`lon`) | float | Bridge location | Decoded from packed `DDMMSSss`/`DDDMMSSss` integers |
| `DECK_COND_058`, `SUPERSTRUCTURE_COND_059`, `SUBSTRUCTURE_COND_060`, `CULVERT_COND_062` | categorical | Component condition ratings (culverts use `CULVERT_COND`; other bridges use the other three) | N (not applicable), 0 (failed) to 9 (excellent) |
| `LOWEST_RATING` | int | FHWA's precomputed minimum of the applicable component ratings above | 0 to 9 |
| `BRIDGE_CONDITION` | categorical | FHWA's precomputed overall rollup | G (Good), F (Fair), P (Poor) |

### Target / label

`deteriorated_next_period` (derived, binary): **1** if `LOWEST_RATING` in the *next* annual record is lower than in
the *current* one, **0** otherwise (stable or improved). Built by joining each state's consecutive year files on
`STRUCTURE_NUMBER_008`.

Measured class balance across the full 681,841 row dataset: **4.34% positive** (29,583 deteriorated transitions
versus 652,258 stable or improved). That is roughly a 22 to 1 imbalance, which the modeling stage handles with
class weighting, and which is why the model is judged on F1, recall, and PR-AUC rather than raw accuracy.

### Candidate features

From the bridge inventory: `age`, `ADT_029`, `PERCENT_ADT_TRUCK_109`, `STRUCTURE_KIND_043A`,
`STRUCTURE_TYPE_043B`, `MAIN_UNIT_SPANS_045`, `STRUCTURE_LEN_MT_049`, `DECK_WIDTH_MT_052`,
`TRAFFIC_LANES_ON_028A`, `SCOUR_CRITICAL_113`, `OWNER_022`, `DECK_STRUCTURE_TYPE_107`,
`INSPECT_FREQ_MONTHS_091`, `STATE`, `lat`, `lon`, and the bridge's **current** condition ratings
(`DECK_COND_058`, `SUPERSTRUCTURE_COND_059`, `SUBSTRUCTURE_COND_060`, `CULVERT_COND_062`, `LOWEST_RATING`). A bridge
already in worse shape is a reasonable prior for predicting further decline, so these are legitimate predictors of
the *next* transition, not leakage from it.

From the weather data: `tmax_mean_c`, `tmin_mean_c`, `tavg_mean_c`, `prcp_total_mm`, and `freeze_thaw_days`
(research question 3).

From the spending data: `bridge_repair_per_bridge_thousands`, the state's bridge repair total divided by its number
of inventoried bridges that year (research question 4).

## 4. Methods

- **Model:** Logistic Regression with `class_weight="balanced"`, chosen because its coefficients show directly
  which features raise or lower risk, which the app needs in order to explain each score.
- **Preprocessing:** numeric features have missing values filled with the median and are scaled; categorical
  features have missing values filled with the most common category and are one hot encoded.
- **Train and test split:** time based. The model trains on transitions ending in 2021 through 2024 (543,921 rows)
  and is tested on transitions ending in 2025 (137,920 rows), which it never sees during training. This matches how
  the model would be used: forecasting next year's risk for bridges already being tracked.
- **Evaluation:** F1, precision, recall, PR-AUC, and ROC-AUC on the 2025 holdout, compared against a majority class
  baseline. Research question 4 is answered by comparing the model with and without the spending feature.

The step by step plan for the remaining work is in [`project_plan.md`](project_plan.md). The exploratory analysis
(schema validation across all 42 files, data quality findings, feature distributions, the label construction, and
the weather data) is in [`notebooks/eda.ipynb`](../notebooks/eda.ipynb).
