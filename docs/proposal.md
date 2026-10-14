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

BridgeWatch AI uses sixteen consecutive years of NBI data (2010 to 2025) for seven states to build a model that
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
geographic and climate diversity, each with **16 annual snapshots (2010 to 2025)**, for 112 raw files in total.

**2. [NOAA nClimGrid-Daily](https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily).** NCEI's daily
gridded climate product (daily maximum, minimum, and average temperature and precipitation, at about 5 km
resolution, for the contiguous U.S., 1951 to present), distributed as monthly netCDF files in the public S3 bucket
`noaa-nclimgrid-daily-pds`. Used for research question 3. See the weather section below.

**3. [FHWA Highway Statistics, Table SF-12A](https://www.fhwa.dot.gov/policyinformation/statistics/2024/sf12a.cfm).**
State highway agency capital outlay by improvement type, in thousands of dollars, published annually. Used for
research question 4. See the spending section below.

### Data size

- Raw bridge files: 112 `.txt` files (comma delimited), **about 933 MB** combined, stored in `data/`.
- Model ready bridge dataset: 15 files, `data/processed/bridge_deterioration_dataset_2010.csv.gz` through
  `_2024.csv.gz`, **4 to 5 MB** each (about 77 MB compressed in total). One file per feature year, so each stays
  under GitHub's 25 MB limit for uploads through the web page.
- Weather features: 16 files, `data/weather_annual_by_bridge_2010.csv` through `_2025.csv`, **about 9 MB** each
  (split by year so each file stays small enough to upload easily).
- State bridge spending: `data/sf12a_state_bridge_spending_2020_2024.csv`, **35 rows** (7 states by 5 years).

### Data shape

- Each raw bridge file has one row per bridge: **123 columns** for 2019 to 2025 and 134 to 137 columns for the
  older years, which use FHWA's earlier layout. Row counts range from 5,195 (Maryland, 2010) to 56,951 (Texas,
  2025).
- Combined 2010 to 2025 raw data: **2,147,605 bridge year records** across all 112 files.
- The **model ready dataset**, built by pairing each bridge's consecutive annual records (2010 to 11, 11 to 12, and
  so on through 24 to 25) within each state: **1,985,801 rows by 39 columns**.
- Weather: **2,146,854 rows** (one per bridge per year) by 8 columns.
- Spending: **35 rows** by 7 columns.

### Time period

2010 to 2025 (16 calendar years) for bridges and weather, yielding 15 year over year transitions per state that the
model learns from. Spending covers 2020 to 2024; 2025 has not been published yet, and 2010 to 2019 have not been
collected. Transitions that start in 2020 through 2024 have matching spending data. Earlier ones do not, so the
experiments with spending use only those later transitions.

### What does each row represent?

- In a **raw bridge** file: one row is one bridge structure, as inventoried by that state in that year.
- In the **model ready** dataset: one row is one bridge's **transition** from one annual inspection to the next
  (for example, "Bridge X's 2022 record: did its condition rating drop by 2023?"). The same physical bridge
  contributes up to 15 such rows, one per consecutive year pair it appears in.
- In the **weather** data: one row is one bridge in one calendar year.
- In the **spending** data: one row is one state in one year.

### Working with the older files

FHWA changed its file format over this period, so the older files need some care:

- **Layouts.** The files have 134 columns (2010 to 2015), 135 (2016), 137 (2017 and 2018), or 123 (2019 to 2025).
  Every column the project uses is present in all of them, apart from the two summaries below.
- **Condition summaries.** FHWA only publishes `LOWEST_RATING` and `BRIDGE_CONDITION` from 2019 on. For 2010 to 2018
  they are rebuilt from the four component ratings, using the same rule (`LOWEST_RATING` is the lowest component
  rating that applies, and `BRIDGE_CONDITION` is Good for 7 to 9, Fair for 5 or 6, and Poor for 4 or below). The rule
  reproduces FHWA's published values exactly on all 959,519 records from 2019 to 2025. Where an older file leaves a
  component rating blank, it is treated as `N` (not applicable), the value later files use.
- **Encoding.** At least one file (`TX18.txt`) is not valid UTF-8, so the files are read as `latin-1`.
- **Bridge numbers.** Michigan's 2010 file numbers its bridges differently from every later year, so only 80 of its
  10,928 bridges match 2011. About 11,000 transitions (0.5% of the total) are therefore missing.
- **Consistency across years.** The code columns (material, owner, scour rating, structure and deck type, condition
  ratings) use the same codes in about the same proportions before and after the 2019 format change, so older and
  newer bridges can be modeled together. One exception is the truck share of traffic (`PERCENT_ADT_TRUCK_109`), which
  is missing for about 13% of bridges in 2010 to 2017 and 1.7% from 2018 on. The gap is almost entirely Texas, where
  about 30% of bridges have no truck share before 2018. The model fills missing values with the median.
- **Duplicates.** 46 rows in 8 files were dropped because the same structure number appeared twice with different
  records and there is no way to tell which is right.
- **Coordinates.** Latitude and longitude are checked against each state's bounding box, since some files hold
  placeholder values (Maryland's 2010 to 2016 files especially). A coordinate that is missing in one year is filled
  in from the same bridge's other years. After that, 595 of the 1,985,801 rows have no usable coordinate.

### Weather data

Pulled for the same 2010 to 2025 period and 7 state footprint as the bridge data. Each bridge is matched to its
nearest nClimGrid-Daily grid cell using its decoded `lat` and `lon` (from the latest year in which it has a valid
coordinate), and daily values are summarized into one row per bridge per year by
[`scripts/fetch_nclimgrid_weather.py`](../scripts/fetch_nclimgrid_weather.py). The raw netCDF grids (roughly 12 GB
across 192 monthly files) are downloaded and discarded by the script; only the small summarized output is stored
in the repo.

| Column | Type | Definition | Values |
|---|---|---|---|
| `tmax_mean_c` | float | Annual mean of daily maximum temperature | °C |
| `tmin_mean_c` | float | Annual mean of daily minimum temperature | °C |
| `tavg_mean_c` | float | Annual mean of daily average temperature | °C |
| `prcp_total_mm` | float | Annual total precipitation | mm, 0 or more |
| `freeze_thaw_days` | int | Count of days where `tmax > 0°C` and `tmin < 0°C` | 0 to 365 |

Joins onto the bridge dataset on `STRUCTURE_NUMBER_008` plus `STATE` plus year. The state is part of the key because
a structure number is only unique within a state (in 2010, one number belongs to both a Michigan and a Texas
bridge). About 1.4% of bridges have all five weather values blank, because their nearest grid cell falls just
offshore or outside the grid's land coverage. They are left blank, not filled with zeros.

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
the same values. Transitions that start before 2020 have no spending data.

Two cautions shape how this data is used. First, raw dollars mostly reflect state size (Texas has about ten times
as many bridges as Maryland), so the model uses spending per inventoried bridge rather than the raw total.
Second, states tend to spend more where bridges are already in poor condition, so a link between spending and
deterioration in this data should not be read as spending causing it.

### Data dictionary (bridge inventory)

Full field definitions are in FHWA's Recording and Coding Guide (linked above). The bridge columns used in this
project are:

| Column | Type | Definition | Values |
|---|---|---|---|
| `STRUCTURE_NUMBER_008` | string | Bridge identifier (join key across years, unique within a state) | Free form, state assigned |
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
| `LOWEST_RATING` | int | Minimum of the applicable component ratings above (published by FHWA from 2019, rebuilt for 2010 to 2018) | 0 to 9 |
| `BRIDGE_CONDITION` | categorical | Overall rollup (published by FHWA from 2019, rebuilt for 2010 to 2018) | G (Good), F (Fair), P (Poor) |

### Target / label

`deteriorated_next_period` (derived, binary): **1** if `LOWEST_RATING` in the *next* annual record is lower than in
the *current* one, **0** otherwise (stable or improved). Built by joining each state's consecutive year files on
`STRUCTURE_NUMBER_008`.

Measured class balance across the full 1,985,801 row dataset: **5.25% positive** (104,201 deteriorated transitions
versus 1,881,600 stable or improved). That is roughly an 18 to 1 imbalance, which the modeling stage handles with
class weighting, and which is why the model is judged on ROC AUC, average precision, and how many of the bridges that
dropped it finds near the top of its ranking, rather than raw accuracy.

The rate is not steady across states and years. California's is 23% and 25% for transitions ending in 2017 and 2018
(against 3 to 7% in most other years), after about a quarter of its ratings went up in 2016. That looks like a change
in how ratings were assigned and not real wear, so the final model leaves out California transitions ending in 2016
to 2020. A few other state and year combinations are also unusually low (Michigan 2011 and 2012, Washington 2011
and 2021).

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

From the spending data: `repair_per_bridge_thousands`, the state's bridge repair total divided by its number
of inventoried bridges that year (research question 4). It was tested and then dropped from the final model, because
it made the predictions worse.

## 4. Methods

- **Model:** Logistic Regression with `class_weight="balanced"`, chosen because its coefficients show directly
  which features raise or lower risk, which the app needs in order to explain each score.
- **Preprocessing:** numeric features have missing values filled with the median and are scaled; categorical
  features have missing values filled with the most common category and are one hot encoded.
- **Train and test split:** time based. The model trains on transitions ending before 2025 and is tested on
  transitions ending in 2025 (137,920 rows, 6,268 of them deteriorations), which it never sees during training. This
  matches how the model would be used: forecasting next year's risk for bridges already being tracked.
- **Training history:** the spending experiments use only the 2020 and later transitions, since spending exists from
  2020. For the final model, three choices of training data are compared on a 2024 validation year (training on
  transitions ending before 2024): all years, all years without California's 2016 to 2020 transitions, and 2020 on
  only. Using a validation year keeps the 2025 test set out of the decision. The choice without California's swing
  years is used (1,720,718 training rows).
- **Scores and risk tiers:** `class_weight="balanced"` inflates the model's raw scores (they average 44% when 4.5% of
  test bridges really dropped), so the scores are corrected with a fixed shift in the log odds, equal to the log odds of a
  drop in the training data. This does not change the ranking. The app groups bridges into three tiers by the corrected
  chance: High is the top 10% of the 2025 test bridges, Medium the next 20%, and Low the rest.
- **Evaluation:** ROC AUC, average precision (area under the precision recall curve), precision and recall, and the
  capture rate (the share of the bridges that dropped that are found in the top 5%, 10%, 20% and so on of the risk
  ranking), on the 2025 test set. A random ranking would score a ROC AUC of 0.5 and an average precision equal to
  the positive rate (4.5%). Differences between model versions are checked with a bootstrap: the test set is redrawn
  at random 200 times, and a difference counts as real only if the middle 95% of the redraws stays on one side of
  zero. Research question 4 is answered by comparing the model with and without the spending feature. Research
  question 3 is answered by comparing it with and without the weather features.

The step by step plan for the remaining work is in [`project_plan.md`](project_plan.md). The exploratory analysis
(schema validation across all 112 files, data quality findings, feature distributions, the label construction, and
the weather and spending data) is in [`notebooks/eda.ipynb`](../notebooks/eda.ipynb), and the model training is in
[`notebooks/model_training_step_10.ipynb`](../notebooks/model_training_step_10.ipynb).
