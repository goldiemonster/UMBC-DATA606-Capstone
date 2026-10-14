# Application

This folder holds the BridgeWatch AI Streamlit app. The pages are not built yet, but the model they need is trained and
saved, and the helper code for scoring bridges is ready.

## What is here

| File | What it is |
|---|---|
| `model/deterioration_model.joblib` | The trained model, saved by [`../notebooks/model_training_step_10.ipynb`](../notebooks/model_training_step_10.ipynb). It holds the fitted pipeline, the feature lists, the probability correction, the risk tier cutoffs, the test scores, and the library versions it was trained with |
| `scoring.py` | Helper functions the app calls: `load_model`, `score_bridges` (estimated chance and risk tier for each bridge), and `explain_bridge` (the biggest reasons behind one bridge's score) |

The saved model was trained with scikit-learn 1.9.1. If the app reports that the model will not load, or warns about a
version mismatch, re-run the step 10 notebook to save the model again with the version installed on your machine.

## What the app will do

- **Bridge Lookup:** pick a state and a bridge to see its estimated chance that its condition rating drops by the
  next inspection, its risk tier, its key attributes (age, material, traffic, current condition), and the factors that
  raise or lower its score. The state's recent bridge repair spending per bridge will be shown alongside as context.
- **State Risk Map:** a map of a state's bridges, colored by risk tier.

How to read the score. It is the estimated chance that the bridge's lowest condition rating drops by one or more points
at its next inspection, not the chance that the bridge becomes unsafe. A drop from 9 to 8 counts the same as a drop from
4 to 3. The High tier (the top 10% of bridges) dropped 14.1% of the time in the 2025 test year, against 7.0% for Medium and
2.5% for Low. High does not mean the bridge is in bad shape: the bridges most likely to drop are the ones rated 8 or 9,
which have the furthest to fall, and most bridges rated Poor land in the Low tier. Each bridge's current condition will be
shown next to its tier for that reason.

## Planned layout

```
app/
  app.py                      home page
  utils.py                    loads the data, caches the model, and wraps the scoring helpers
  scoring.py                  scoring and explanation helpers (done)
  model/
    deterioration_model.joblib    trained model (done)
  pages/
    1_Bridge_Lookup.py
    2_State_Risk_Map.py
```

## Running it (once built)

```
pip install -r requirements.txt
streamlit run app/app.py
```

The app scores each bridge's most recent (2025) record. That record is built ahead of time by
`scripts/build_current_risk_snapshot.py`, which joins the 2025 inventory with the 2025 weather file and writes
`data/processed/bridge_current_risk_input.csv.gz`, so the app never has to read the raw inventory files. The final model
uses bridge and weather features only, so scoring does not need spending data. Spending figures for 2025 are not published
yet, so the spending shown as context will be each state's 2024 figure, labelled that way.
