# Application

This folder will hold the BridgeWatch AI Streamlit app. It has not been built yet. The model it depends on is
trained (see [`../notebooks/model_training_step_9.ipynb`](../notebooks/model_training_step_9.ipynb)) but has not been
saved yet; that is the next step in [`../docs/project_plan.md`](../docs/project_plan.md).

## What the app will do

- **Bridge Lookup:** pick a state and a bridge to see its predicted chance that its condition rating drops by the
  next inspection, its key attributes (age, material, traffic, current condition), and the factors that raise or
  lower its score. The state's recent bridge repair spending per bridge will be shown alongside as context.
- **State Risk Map:** a map of a state's bridges, colored by predicted risk.

The score is the chance that the bridge's lowest condition rating drops by one or more points, not the chance that
the bridge becomes unsafe. A drop from 9 to 8 counts the same as a drop from 4 to 3, so the app will say so.

## Planned layout

```
app/
  app.py                      home page
  utils.py                    loads the model and data, scores bridges, explains scores
  model/
    deterioration_model.joblib    trained model, saved by the model training notebook
  pages/
    1_Bridge_Lookup.py
    2_State_Risk_Map.py
```

## Running it (once built)

```
pip install -r requirements.txt
streamlit run app/app.py
```

The app will score each bridge's most recent (2025) record. That record is built ahead of time by
`scripts/build_current_risk_snapshot.py`, which joins the 2025 inventory with the 2025 weather file, so the app never
has to read the raw inventory files. The final model uses bridge and weather features only, so scoring does not need
spending data. Spending figures for 2025 are not published yet, so the spending shown as context will be each state's
2024 figure, labelled that way.
