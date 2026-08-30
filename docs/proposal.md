# BridgeWatch AI: Predicting Bridge Condition Deterioration from Federal Inspection Records

## 1. Title and Author

- **Project Title:** BridgeWatch AI — Predicting Bridge Condition Deterioration from Federal Inspection Records
- Prepared for the UMBC Data Science Master's Degree Capstone by Dr. Chaojie (Jay) Wang
- **Author:** Edmund L. Goldsberry
- **GitHub repository:** https://github.com/goldiemonster/UMBC-DATA606-Capstone
- **LinkedIn profile:** _TBD — to be added_
- **PowerPoint presentation:** _TBD — added when the presentation is complete_
- **YouTube video:** _TBD — added when the presentation is complete_

## 2. Background

### What is it about?

Every U.S. highway bridge is inspected on a regular cycle (generally every 12–24 months) under the National
Bridge Inspection Standards, and the results are reported to the Federal Highway Administration (FHWA) as
part of the **National Bridge Inventory (NBI)** — a structured record, for every bridge in the country, of its
physical characteristics (age, material, span, traffic volume) and its condition ratings (deck, superstructure,
substructure, and an overall condition category of Good/Fair/Poor).

BridgeWatch AI uses six consecutive years of NBI data (2020–2025) for seven states to build a model that
predicts whether a bridge's condition rating will **decline by its next inspection cycle** — before that decline
shows up in an inspection report. The end product is a Streamlit application that lets a user look up a bridge
(or browse a state map of bridges color-coded by predicted risk) and see a deterioration-risk score along with
the factors driving it.

### Why does it matter?

The 2021 Infrastructure Investment and Jobs Act highlighted just how much of the U.S. bridge inventory is aging
past its intended design life, and bridge maintenance budgets are finite: state DOTs cannot re-inspect or
rehabilitate every structure every year. A model that flags which bridges are statistically likely to deteriorate
soon — rather than waiting for the next scheduled inspection to find out — gives inspection planners a
data-driven way to prioritize limited inspection and maintenance resources toward the structures most likely to
need it, instead of relying solely on fixed inspection intervals and age-based heuristics.

### Research questions

1. Can machine learning predict whether a bridge's condition rating will decline by its next inspection cycle,
   using only information available at the time of the current inspection (structural characteristics, traffic
   loading, age, and current condition)?
2. Which factors are the strongest predictors of near-term deterioration — age, traffic volume/truck loading,
   construction material, scour vulnerability, or time since last inspection/rehabilitation?
3. Does deterioration risk vary meaningfully by state/region, suggesting climate or maintenance-practice effects
   beyond what's captured in the bridge's own attributes?

## 3. Data

### Data sources

[FHWA National Bridge Inventory](https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm) — the official, publicly
published ASCII bridge inventory files FHWA releases annually for every U.S. state, territory, and federal
agency, submitted under the National Bridge Inspection Standards (23 CFR 650 Subpart C). Field definitions
follow FHWA's *Recording and Coding Guide for the Structure Inventory and Appraisal of the Nation's Bridges*
(each column name embeds its official NBI item number, e.g. `YEAR_BUILT_027` = Item 27).

This project uses **7 states** (California, Florida, Maryland, Michigan, New York, Texas, Washington) chosen
for geographic/climate diversity, each with **6 annual snapshots (2020–2025)** — 42 raw files in total.

### Data size

- Raw files: 42 `.txt` files (comma-delimited), **~326 MB** combined, stored in `data/`.
- Model-ready dataset built from them: `data/processed/bridge_deterioration_dataset.csv.gz` (**~27 MB**
  compressed; ~118 MB uncompressed — compressed to stay under GitHub's 100 MB per-file limit).

### Data shape

- Each raw file: **123 columns**, one row per bridge. Row counts range from 5,430 (Maryland, 2020) to 56,951
  (Texas, 2025).
- Combined 2020–2025 raw data: **824,312 bridge-year records** across all 42 files.
- The **model-ready dataset**, built by pairing each bridge's consecutive annual records (2020→21, 21→22, …,
  24→25) within each state: **681,841 rows × 39 columns**.

### Time period

2020–2025 (6 calendar years), yielding 5 year-over-year transitions per state that the model learns from.

### What does each row represent?

- In a **raw** file: one row = one bridge structure, as inventoried by that state in that year.
- In the **model-ready** dataset: one row = one bridge's **transition** from one annual inspection to the next
  (e.g., "Bridge X's 2022 record → did its condition rating drop by 2023?"). The same physical bridge
  contributes up to 5 such rows (one per consecutive year-pair it appears in).

### Data dictionary

Full field definitions are in FHWA's Recording and Coding Guide (linked above); the columns used in this
project are:

| Column | Type | Definition | Values |
|---|---|---|---|
| `STRUCTURE_NUMBER_008` | string | Unique bridge identifier (join key across years) | Free-form, state-assigned |
| `STATE` (derived) | categorical | Two-letter state code | CA, FL, MD, MI, NY, TX, WA |
| `YEAR_BUILT_027` | int | Year the bridge was originally built | e.g. 1937–2025 |
| `age` (derived) | int | Inspection year − `YEAR_BUILT_027` | 0–225 |
| `ADT_029` | int | Average Daily Traffic | 0–810,110 |
| `PERCENT_ADT_TRUCK_109` | float | % of ADT that is truck traffic | 0–100 |
| `STRUCTURE_KIND_043A` | categorical | Primary material/design (NBI Item 43A) | 0=Other, 1/2=Concrete (simple/cont.), 3/4=Steel (simple/cont.), 5/6=Prestressed concrete (simple/cont.), 7=Wood/timber, 8=Masonry, 9=Aluminum/iron |
| `STRUCTURE_TYPE_043B` | categorical | Structure type/design (NBI Item 43B) | Coded per Recording & Coding Guide |
| `MAIN_UNIT_SPANS_045` | int | Number of spans in main unit | 1+ |
| `STRUCTURE_LEN_MT_049` | float | Total structure length (meters) | > 0 |
| `DECK_WIDTH_MT_052` | float | Deck width (meters) | > 0 |
| `TRAFFIC_LANES_ON_028A` | int | Number of traffic lanes on the structure | 0+ |
| `SCOUR_CRITICAL_113` | categorical | Scour vulnerability rating | N, U, 0–9 |
| `OWNER_022` | categorical | Owning agency type | Coded per Recording & Coding Guide |
| `INSPECT_FREQ_MONTHS_091` | int | Months between required inspections | Typically 12–48 |
| `LAT_016` / `LONG_017` (decoded to `lat`/`lon`) | float | Bridge location | Decoded from packed `DDMMSSss`/`DDDMMSSss` integers |
| `DECK_COND_058`, `SUPERSTRUCTURE_COND_059`, `SUBSTRUCTURE_COND_060`, `CULVERT_COND_062` | categorical | Component condition ratings (mutually exclusive: culvert bridges use `CULVERT_COND`; others use the other three) | N (n/a), 0 (failed) – 9 (excellent) |
| `LOWEST_RATING` | int | FHWA's precomputed minimum of the applicable component ratings above | 0–9 |
| `BRIDGE_CONDITION` | categorical | FHWA's precomputed overall rollup | G (Good), F (Fair), P (Poor) |

### Target / label

`deteriorated_next_period` (derived, binary): **1** if `LOWEST_RATING` in the *next* annual record is lower than
in the *current* one, **0** otherwise (stable or improved). Built by joining each state's consecutive-year files
on `STRUCTURE_NUMBER_008`.

Measured class balance across the full 681,841-row dataset: **4.34% positive** (29,583 deteriorated
transitions vs. 652,258 stable/improved) — a ~22:1 imbalance that the modeling stage will need to account for
(class weighting/resampling; F1, recall, and PR-AUC rather than raw accuracy).

### Candidate features

`age`, `ADT_029`, `PERCENT_ADT_TRUCK_109`, `STRUCTURE_KIND_043A`, `STRUCTURE_TYPE_043B`,
`MAIN_UNIT_SPANS_045`, `STRUCTURE_LEN_MT_049`, `DECK_WIDTH_MT_052`, `TRAFFIC_LANES_ON_028A`,
`SCOUR_CRITICAL_113`, `OWNER_022`, `INSPECT_FREQ_MONTHS_091`, `STATE`/`COUNTY_CODE_003`, `lat`/`lon`, and
the bridge's **current** condition ratings (`DECK_COND_058`, `SUPERSTRUCTURE_COND_059`,
`SUBSTRUCTURE_COND_060`, `LOWEST_RATING`) — a bridge already in worse shape is a reasonable prior for
predicting further decline, so these are legitimate predictors of the *next* transition, not leakage from it.

Full exploratory analysis — schema validation across all 42 files, data-quality findings (duplicate records,
malformed coordinate fields, missing-value patterns), feature distributions, and the label-construction
pipeline — is in [`notebooks/eda.ipynb`](../notebooks/eda.ipynb).
