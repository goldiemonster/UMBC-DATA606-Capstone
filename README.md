# BridgeWatch AI

Predicting bridge condition deterioration from federal inspection records, for the UMBC DATA 606 Capstone.

**Author:** Edmund L. Goldsberry
**Prepared for:** UMBC Data Science Master's Degree Capstone, Dr. Chaojie (Jay) Wang
**Repository:** https://github.com/goldiemonster/UMBC-DATA606-Capstone

## What this project does

Every U.S. highway bridge is inspected on a regular cycle under the National Bridge Inspection Standards, and the
results are reported to the Federal Highway Administration as part of the National Bridge Inventory (NBI). This
project uses six years of NBI records (2020 to 2025) across seven states (California, Florida, Maryland, Michigan,
New York, Texas, Washington) to predict whether a bridge's condition rating will decline by its next inspection.
The bridge records are combined with NOAA weather data for each bridge's location and with each state's annual
spending on bridge repair. The end goal is a Streamlit app that lets a user look up a bridge or browse a state map
of bridges color coded by predicted risk, and see the factors driving that score.

Four research questions guide the project:

1. Can machine learning predict whether a bridge's condition rating will decline by its next inspection cycle,
   using only information available at the time of the current inspection?
2. Which factors are the strongest predictors of near term deterioration: age, traffic volume, construction
   material, scour vulnerability, or time since last inspection?
3. Does deterioration risk vary by state or region in a way that suggests climate effects, separate from a
   bridge's own attributes?
4. Is a state's spending on bridge repair related to how quickly its bridges deteriorate, and does including it
   improve predictions?

See [`docs/proposal.md`](docs/proposal.md) for the full proposal and data dictionary, and
[`docs/project_plan.md`](docs/project_plan.md) for the plan for the remaining work.

## Project status

- **Data:** done. Bridge inventory, weather, and state bridge spending are all cleaned and in [`data/`](data).
- **Exploratory data analysis:** done for the bridge and weather data, in [`notebooks/eda.ipynb`](notebooks/eda.ipynb).
  The spending data has not been explored there yet.
- **Model training:** in progress. Steps 1 to 4 (load and join data, define features, split, fit) are done, in
  [`notebooks/`](notebooks). Next up: adding the spending feature, then evaluation.
- **Streamlit app:** not started yet. Will live in [`app/`](app).

## Data

| Source | What it adds | Years |
|---|---|---|
| FHWA National Bridge Inventory | Each bridge's age, design, traffic, and condition ratings | 2020 to 2025 |
| NOAA nClimGrid-Daily | Annual temperature, precipitation, and freeze and thaw days at each bridge's location | 2020 to 2025 |
| FHWA Highway Statistics, Table SF-12A | Each state's annual spending on bridge replacement and rehabilitation | 2020 to 2024 |

See [`data/README.md`](data/README.md) for what each file contains and how it was built.

## Repository structure

```
data/         Raw bridge files, weather data, state spending data, and the processed model ready dataset
notebooks/    EDA and model training notebooks
scripts/      Standalone data pipeline scripts (weather fetch, current risk snapshot)
app/          Streamlit application (not built yet)
docs/         Proposal, project plan, and presentation
```

Each folder has its own README with more detail: [`data/README.md`](data/README.md),
[`notebooks/README.md`](notebooks/README.md), [`app/README.md`](app/README.md),
[`docs/README.md`](docs/README.md).

## Getting started

Install the packages the notebooks and app use:

```
pip install -r requirements.txt
```

Then open the notebooks in order:

1. [`notebooks/eda.ipynb`](notebooks/eda.ipynb): builds the model ready bridge dataset, walks through the data
   quality checks behind it, and explores the weather data.
2. The model training notebooks, newest first (`model_training_step_4.ipynb` currently contains every step so
   far). They load their data straight from this GitHub repository, so they run without downloading anything.

Once the Streamlit app is built, it will run locally with `streamlit run app/app.py`.
