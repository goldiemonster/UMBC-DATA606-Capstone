# BridgeWatch AI

Predicting bridge condition deterioration from federal inspection records, for the UMBC DATA 606 Capstone.

**Author:** Edmund L. Goldsberry
**Prepared for:** UMBC Data Science Master's Degree Capstone, Dr. Chaojie (Jay) Wang
**Repository:** https://github.com/goldiemonster/UMBC-DATA606-Capstone

## What this project does

Every U.S. highway bridge is inspected on a regular cycle under the National Bridge Inspection Standards, and
the results are reported to the Federal Highway Administration as part of the National Bridge Inventory (NBI).
This project uses six years of NBI records (2020 to 2025) across seven states (California, Florida, Maryland,
Michigan, New York, Texas, Washington), combined with NOAA weather data, to predict whether a bridge's
condition rating will decline by its next inspection. The end goal is a Streamlit app that lets a user look up
a bridge or browse a state map of bridges color coded by predicted risk, and see the factors driving that
score.

Three research questions guide the project:

1. Can machine learning predict whether a bridge's condition rating will decline by its next inspection cycle,
   using only information available at the time of the current inspection?
2. Which factors are the strongest predictors of near term deterioration: age, traffic volume, construction
   material, scour vulnerability, or time since last inspection?
3. Does deterioration risk vary by state or region in a way that suggests climate or maintenance practice
   effects, separate from a bridge's own attributes?

See [`docs/proposal.md`](docs/proposal.md) for the full proposal, data dictionary, and research background.

## Project status

- Data pipeline: done. Raw NBI files and NOAA weather features are pulled together into a model ready dataset.
- Exploratory data analysis: done, in [`notebooks/eda.ipynb`](notebooks/eda.ipynb).
- Model training: in progress, in [`notebooks/`](notebooks) (built step by step, one function at a time).
- Streamlit app: not started yet. Will live in [`app/`](app).

## Repository structure

```
data/         Raw NBI files, NOAA weather data, and the processed model ready datasets
notebooks/    EDA and model training notebooks
scripts/      Standalone data pipeline scripts (weather fetch, current risk snapshot)
app/          Streamlit application (in progress)
docs/         Project proposal, data dictionary, and supporting materials
```

Each of these folders has its own README with more detail: [`data/README.md`](data/README.md),
[`notebooks/README.md`](notebooks/README.md), [`app/README.md`](app/README.md).

## Data

- **Bridge inventory:** FHWA National Bridge Inventory, 7 states, 6 annual snapshots (2020 to 2025). See
  [`data/README.md`](data/README.md) for the raw file layout and the processed dataset's columns.
- **Weather:** NOAA nClimGrid-Daily, aggregated to annual temperature and precipitation summaries per bridge,
  covering the same 2020 to 2025 period. Also documented in [`data/README.md`](data/README.md).

## Getting started

Clone the repository and open the notebooks in order:

1. [`notebooks/eda.ipynb`](notebooks/eda.ipynb): builds the processed bridge dataset and the weather features,
   and walks through the data quality checks behind them.
2. `notebooks/model_training.ipynb`: trains the deterioration prediction model (in progress).

Once the Streamlit app is built, it will be run locally with `streamlit run app/app.py`.
