# BridgeWatch AI

Predicting bridge condition deterioration from federal inspection records, for the UMBC DATA 606 Capstone.

**Author:** Edmund L. Goldsberry
**Prepared for:** UMBC Data Science Master's Degree Capstone, Dr. Chaojie (Jay) Wang
**Repository:** https://github.com/goldiemonster/UMBC-DATA606-Capstone

## What this project does

Every U.S. highway bridge is inspected on a regular cycle under the National Bridge Inspection Standards, and the
results are reported to the Federal Highway Administration as part of the National Bridge Inventory (NBI). This
project uses sixteen years of NBI records (2010 to 2025) across seven states (California, Florida, Maryland,
Michigan, New York, Texas, Washington) to predict whether a bridge's condition rating will decline by its next
inspection. The bridge records are combined with NOAA weather data for each bridge's location and with each state's
annual spending on bridge repair. The end goal is a Streamlit app that lets a user look up a bridge or browse a state
map of bridges color coded by predicted risk, and see the factors driving that score.

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

- **Data:** done. Bridge inventory and weather for 2010 to 2025, and state bridge spending for 2020 to 2024, are all
  cleaned and in [`data/`](data).
- **Exploratory data analysis:** done for all three data sources, in [`notebooks/eda.ipynb`](notebooks/eda.ipynb).
- **Model training:** done, in [`notebooks/`](notebooks). The newest notebook,
  [`model_training_step_11.ipynb`](notebooks/model_training_step_11.ipynb), has every step on the full 2010 to 2025
  history. It saves two models to `app/model/`: one for the chance that a bridge's rating drops, and one for the chance
  that a bridge falls into Poor condition.
- **Current risk snapshot:** done. Each bridge's 2025 record with 2025 weather, ready for the app to score.
- **Streamlit app:** done and tested, in [`app/`](app). It has a home page, a Bridge Lookup page, and a State Risk Map page,
  and lets the user switch between the two views of risk.

## What the model has found so far

- A bridge's current condition is by far the strongest predictor of whether its rating will drop, followed by its
  age. Traffic, size, and weather matter much less (research question 2).
- On the 2025 test year, the final model has a ROC AUC of 0.726. Ranking bridges by its risk score, the top 20% of
  bridges contain about half (50.3%) of the bridges whose rating actually dropped (research question 1).
- Weather makes no clear difference to the predictions once the bridge's own features and state are known
  (research question 3).
- State repair spending made the predictions worse, so it is not used to score bridges (research question 4). It
  can still be shown in the app as background about each state.
- The older years (2010 to 2019) help, but only after leaving out California's 2016 to 2020 transitions, when its
  ratings swung sharply in a way that does not look like real wear.
- The model's raw scores run far too high as probabilities, so they are corrected before the app shows them. A bridge in
  the High risk tier (the top 10%) dropped 14.1% of the time in 2025, against 2.5% for the Low tier. High means a rating
  drop is likely, not that the bridge is in bad shape: the bridges most likely to drop are the ones rated 8 or 9.
- That is why the app has a second model, for bridges that are not Poor yet and may fall into Poor (a lowest rating of 4
  or below). It is far better at that question (ROC AUC 0.89, against 0.55 for the first model). Its High tier is
  almost entirely bridges rated 5, holds 59% of the bridges that fell into Poor in 2025, and dropped into Poor 3.25% of
  the time, against 0.05% for the Low tier.

## Data

| Source | What it adds | Years |
|---|---|---|
| FHWA National Bridge Inventory | Each bridge's age, design, traffic, and condition ratings | 2010 to 2025 |
| NOAA nClimGrid-Daily | Annual temperature, precipitation, and freeze and thaw days at each bridge's location | 2010 to 2025 |
| FHWA Highway Statistics, Table SF-12A | Each state's annual spending on bridge replacement and rehabilitation | 2020 to 2024 |

See [`data/README.md`](data/README.md) for what each file contains and how it was built.

## Repository structure

```
data/         Raw bridge files, weather data, state spending data, and the processed model ready dataset
notebooks/    EDA and model training notebooks
scripts/      Standalone data pipeline scripts (weather fetch, current risk snapshot)
app/          Streamlit application and the two saved models
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
   quality checks behind it, and explores the weather and state spending data.
2. [`notebooks/model_training_step_10.ipynb`](notebooks/model_training_step_10.ipynb): the complete model training
   notebook. It loads its data straight from this GitHub repository, so it runs without downloading anything. The
   earlier model training notebooks show how it was built up one step at a time (see
   [`notebooks/README.md`](notebooks/README.md)).

Then run the app, from the project folder:

```
streamlit run app/app.py
```

It opens in your browser. The app reads the saved models in `app/model/` and the snapshot in `data/processed/`, so it
needs no internet connection apart from the street map background on the two map views (a Map background choice in the
sidebar offers OpenStreetMap, CARTO, or a Plain background for working offline). See [`app/README.md`](app/README.md) for what each page does and how to read the scores.
