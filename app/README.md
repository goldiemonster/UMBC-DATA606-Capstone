# Application

The BridgeWatch AI Streamlit app. It lets you look up a single bridge or see a whole state on a map, and it estimates
each bridge's risk in two ways.

## Running it

From the project folder:

```
pip install -r requirements.txt
streamlit run app/app.py
```

It opens in your browser. The app reads the two saved models in `app/model/` and the 2025 bridge snapshot in
`data/processed/`, so it needs no other downloads. The one thing that uses the internet is the street map behind the bridges
on the State Risk Map and Bridge Lookup pages. Use the **Map background** choice in the sidebar to pick OpenStreetMap
or CARTO street maps, or **Plain** to draw only the bridges, for example when working offline or when your network blocks a
map provider.

## Two views of risk

A **Risk view** choice in the sidebar switches every page between two questions. Each one has its own trained model.

| View | The question | Which bridges it scores |
|---|---|---|
| Chance of any rating drop | Will this bridge's lowest condition rating drop by its next inspection? | All bridges |
| Chance of falling into Poor condition | Will this bridge fall into Poor condition (a lowest rating of 4 or below) by its next inspection? | Bridges that are not Poor now. Bridges that already are Poor show as "Already Poor" |

Each view groups bridges into three tiers. High is the top 10% of bridges in the 2025 test year, Medium is the next 20%, and Low is
the rest. The tiers earned their names: in 2025, 14.1% of High tier bridges had a rating drop (against 2.5% for Low), and 3.25% of
High tier bridges fell into Poor (against 0.05% for Low).

How to read them:

- The chance is an estimate. The models are better at ranking bridges than at predicting any single bridge.
- A rating drop is not a failure. A bridge going from 9 to 8 counts the same as one going from 4 to 3, so the High tier in the first
  view is mostly bridges rated 8 or 9, which have the furthest to fall. It does not mean a bridge is in bad shape.
- The second view points at bridges closer to the line. Its High tier is almost entirely bridges rated 5.
- Poor is the federal term for a lowest rating of 4 or below. It does not mean unsafe.
- The app does not predict collapse and does not tell anyone to close a bridge.

## The pages

- **Home:** what the app does, the two views side by side with how each tier really performed in 2025, and how each state's
  bridges split across the tiers.
- **Bridge Lookup:** pick a state, then search by structure number, road, or place. It shows both models' estimated chance and
  tier, the bridge's key facts (including its current condition), a chart of the biggest reasons behind the score for the
  selected view, the state's 2024 repair spending per bridge as background, and a small map.
- **State Risk Map:** a state's bridges colored by tier for the selected view. You can filter by tier, limit how many bridges are
  drawn (all High and Medium tier bridges are always kept, and the rest are a random sample for large states), and see the 20
  highest scoring bridges in the state.

## What is in this folder

```
app/
  app.py                      starts the app and sets up the pages
  screens/
    home.py                   the home page
    bridge_lookup.py
    state_risk_map.py
  utils.py                    loads the data and models, scores the bridges, and formats text
  scoring.py                  scoring and explanation helpers
  model/
    deterioration_model.joblib    the model for any rating drop
    poor_condition_model.joblib   the model for falling into Poor
```

The two model files are saved by [`../notebooks/model_training_step_11.ipynb`](../notebooks/model_training_step_11.ipynb), which
also tests `scoring.py` against the notebook's own results. Each file holds the fitted model, its feature lists, the correction
that turns raw scores into chances, the tier cutoffs, how each tier performed in 2025, its test scores, and the library versions it
was trained with.

The saved models were trained with scikit-learn 1.9.1. If the app cannot load them on your machine, or warns about a version
mismatch, run the step 11 notebook again to save the models with the version installed there.

## Where the data comes from

The app scores each bridge's most recent (2025) record. That table is built ahead of time by
`scripts/build_current_risk_snapshot.py`, which joins the 2025 inventory with the 2025 weather file and writes
`data/processed/bridge_current_risk_input.csv.gz`, so the app never reads the raw inventory files. The models use bridge and
weather features only. State spending was tested as a predictor and made the predictions worse, so it is not used in either score. It
is shown only as background on the Bridge Lookup page, using each state's 2024 figure, the most recent one published.
