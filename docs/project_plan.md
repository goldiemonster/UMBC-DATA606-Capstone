# BridgeWatch AI: Project Plan

This plan covers where the project stands and what is left. Both models are trained, saved, and checked, the inputs the
app needs are built, and the Streamlit app is built and tested. What is left is housekeeping, listed at the end. The project uses three data sources: the bridge inventory and weather
data (2010 to 2025) and state bridge repair spending from FHWA Highway Statistics Table SF-12A (2020 to 2024).

## Where the project stands

| Piece | Status |
|---|---|
| Bridge inventory data (NBI, 7 states, 2010 to 2025, 112 files) | Done |
| Model ready bridge dataset (15 yearly files, `bridge_deterioration_dataset_{from_year}.csv.gz`) | Done |
| Weather features (NOAA nClimGrid-Daily, 2010 to 2025, 16 yearly files) | Done, rebuilt with the corrected script |
| State bridge spending (FHWA SF-12A, 2020 to 2024) | Done, cleaned into `data/sf12a_state_bridge_spending_2020_2024.csv`. 2010 to 2019 not collected |
| Exploratory data analysis (`notebooks/eda.ipynb`) | Done for all three data sources, covering 2010 to 2025 |
| Model training, loading and joining, features, split, fit, evaluation | Done |
| Model training, comparing weather and spending, final model, interpretability | Done |
| Model training on the full 2010 to 2025 history (`model_training_step_9.ipynb`) | Done |
| Saving the trained model (`model_training_step_10.ipynb`) | Done, saved to `app/model/deterioration_model.joblib` |
| Second model, chance of falling into Poor (`model_training_step_11.ipynb`) | Done, saved to `app/model/poor_condition_model.joblib` |
| Current risk snapshot for the app | Done, `data/processed/bridge_current_risk_input.csv.gz` (139,079 bridges, 2025 weather) |
| Streamlit app (`app/`) | Done and tested: home page, Bridge Lookup, State Risk Map, and a switch between the two views of risk |

## Research questions

1. Can machine learning predict whether a bridge's condition rating will decline by its next inspection, using
   only information available at the time of the current inspection?
2. Which factors are the strongest predictors of near term deterioration?
3. Does deterioration risk vary by state or region in a way that suggests climate effects, beyond what the
   bridge's own attributes explain?
4. Is a state's bridge repair spending related to how quickly its bridges deteriorate, and does adding it
   improve predictions?

### What the work has found so far

1. **Prediction.** The final model (Logistic Regression on bridge and weather features) scores a ROC AUC of 0.726 on
   the 2025 test year. If inspectors worked down its risk ranking, the top 20% of bridges would contain about half
   (50.3%) of the bridges whose rating actually dropped. The ranking is useful, but not sharp: only about 5% of
   transitions in the data are drops, and a rating can drop for reasons nothing in the data records.
2. **Predictors.** A bridge's current condition is by far the strongest predictor, followed by its age. Traffic,
   size, and weather matter much less. One caution: the label counts any one point drop, so a bridge rated 9 slipping
   to 8 counts the same as one rated 4 slipping to 3, and bridges at the top of the scale drop most often.
3. **Climate.** Weather adds almost nothing once the bridge's own features and its state are known. The change in ROC
   AUC from adding it is +0.0003, which cannot be told apart from luck. States do differ in risk even after
   accounting for everything else (Washington is lowest, California, New York, and Michigan highest).
4. **Spending.** Adding spending per bridge made the predictions worse (a ROC AUC change of -0.0057, with the whole
   95% range below zero), so it is not used to score bridges.

**What the score means.** The model's raw scores run far too high as probabilities (they average 44% when only 4.5% of
bridges really dropped), because balanced class weights inflate them. Step 10 corrects them with one fixed shift in the
log odds, and the corrected chances track the real drop rates closely (for example, 12.9% predicted against 14.1% actual in
the top tenth of bridges). The app shows three tiers. A High tier bridge (the top 10%) dropped 14.1% of the time in 2025,
against 7.0% for Medium and 2.5% for Low, and the High tier holds 31% of all the drops. One caution: High does not mean
"in bad shape". The bridges most likely to drop are the ones rated 8 or 9, which have the furthest to fall, and most
bridges rated Poor land in the Low tier. The app has to say this plainly.

**The second view: falling into Poor.** Because the first view's High tier points at bridges rated 8 or 9, step 11 adds a second
model for the question inspectors and the public are more likely to ask: which bridges that are not Poor yet will fall into Poor
(a lowest rating of 4 or below) by the next inspection? It is a rarer event (0.69% of transitions) and the model is much better
at it: ROC AUC 0.89 on the 2025 test set, against 0.55 for the first model. Its top 20% of bridges contain 84% of the bridges
that fell into Poor, and its High tier (10% of bridges) holds 59% of them, with 3.25% falling into Poor against 0.05% in the Low
tier. The High tier is almost entirely bridges rated 5, ranked by the other facts about them. The two views disagree a lot: only
0.7% of the bridges above Poor are High in both, so the app offers both.

## The three data sources

| Source | Level of detail | Years | Join key |
|---|---|---|---|
| FHWA National Bridge Inventory | One row per bridge per year | 2010 to 2025 | `STRUCTURE_NUMBER_008` |
| NOAA nClimGrid-Daily weather | One row per bridge per year | 2010 to 2025 | `STRUCTURE_NUMBER_008` + `STATE` + year |
| FHWA SF-12A capital outlay | One row per state per year | 2020 to 2024 | `STATE` + year |

The weather data is matched to each bridge's own location, so every bridge gets its own values. The state is part of
the weather join key because a structure number is only unique within a state. The spending data only exists at the
state level, so every bridge in the same state and year gets the same values.

### Decisions made when extending the data to 2010 to 2025

- **Different file layouts.** FHWA changed its format over this period, so the raw files have 134 to 137 columns
  before 2019 and 123 from 2019 on. The columns the project uses are in every file. At least one file (`TX18.txt`)
  is not valid UTF-8, so the files are read as `latin-1`.
- **Rebuilt condition summaries.** FHWA only publishes `LOWEST_RATING` and `BRIDGE_CONDITION` from 2019 on. For 2010
  to 2018 they are rebuilt from the four component ratings. The same rule reproduces FHWA's published values
  exactly on all 959,519 records from 2019 to 2025, and the share of bridges in poor condition shows no jump at the
  2018 to 2019 boundary.
- **Michigan's 2010 file** numbers its bridges differently from every later year, so only 80 of its 10,928 bridges
  match 2011. About 11,000 Michigan transitions (0.5% of the total) are therefore missing.
- **Truck share.** `PERCENT_ADT_TRUCK_109` is missing for about 30% of Texas bridges before 2018 (13% of all bridges
  in 2010 to 2017, against 1.7% from 2018). Other code and numeric columns look the same in every year. The model
  fills the gap with the median.
- **Duplicate structure numbers.** 46 rows in 8 files were dropped because the same number appeared twice with
  different records and there is no way to tell which is right.
- **Missing coordinates.** Maryland's older files have many all-zero placeholder coordinates. A coordinate that is
  missing in one year is filled in from the same bridge's other years, which brings the rows with no usable
  coordinate from 27,552 down to 595 out of 1,985,801.
- **California's 2016 to 2020 ratings.** About a quarter of its ratings went up in 2016, then 23% and 25% of its
  bridges dropped in 2017 and 2018. That looks like a change in how ratings were assigned, not real wear. Training
  on those rows made the model worse, so the final model leaves out California transitions ending 2016 to 2020. The
  rows stay in the processed dataset.
- **Weather fix.** The first version of the weather files stored a missing grid cell as zero precipitation and zero
  freeze and thaw days. They are now left blank, and the files were rebuilt for all 16 years with one script.

### State bridge spending (SF-12A)

FHWA's Table SF-12A reports what each state highway agency actually spent on capital projects, in thousands of
dollars, broken out by improvement type. Each year's workbook has 14 sheets, one for each road system and area
(rural and urban interstates, freeways, arterials, collectors, and local roads). Four columns cover bridge work:
new bridges, bridge replacement, major bridge rehabilitation, and minor bridge rehabilitation.

How the cleaned file was built:

* All 14 sheets in each of the five yearly workbooks (2020 to 2024) were read, and every sheet was confirmed to
  have the same column layout and the same meaning in every year.
* Each row's parsed values were checked against that row's reported total. All 490 state, year, and sheet
  combinations matched.
* The four bridge columns were summed across all 14 sheets for each state and year, then filtered down to the
  seven project states.
* `bridge_repair_total_thousands` is replacement plus major rehabilitation plus minor rehabilitation. New bridge
  construction is kept as its own column but left out of the total, since building a new bridge is not the same
  as maintaining the existing inventory.

Summing all 14 sheets matters. The rural interstate sheet alone accounts for only about 6% of bridge repair
spending across these states. Using it by itself would have made Washington look like it spent nothing on bridge
repair from 2020 to 2023.

## Remaining work

### Part A: model training (done)

The model training notebook was built step by step (see [`../notebooks/README.md`](../notebooks/README.md)), and
`model_training_step_9.ipynb` now holds every step on the full 2010 to 2025 history:

* Loading and joining the bridge, weather, and spending data, from GitHub with a local fallback.
* The 10 categorical and 16 numeric features, the time based split (train on every transition ending before 2025,
  test on those ending in 2025), and the preprocessing and Logistic Regression pipeline.
* Evaluation on the 2025 test set: ROC AUC, average precision, precision and recall, and the capture rate (how many
  of the bridges that dropped fall in the top 5%, 10%, 20% and so on of the risk ranking).
* The comparison with and without weather and spending, with a bootstrap check on each difference.
* The choice of training history, made on a 2024 validation year so the 2025 test set is not used to pick it.
* The final model and its interpretation (coefficients, odds multipliers, and an `explain_bridge()` function whose
  contributions add up exactly to the model's score).

**Step 10: save the model (done, `model_training_step_10.ipynb`).**

* Corrects the raw scores into real probabilities and defines the Low, Medium, and High risk tiers (the cutoffs are a
  corrected chance of 4.7% for Medium and 8.3% for High).
* Saves the final pipeline, its feature lists, the probability shift, the tier cutoffs, its test metrics, its coefficient
  table, a description of the training data, and the library versions to `app/model/deterioration_model.joblib`
  (about 19 KB).
* Tests the helper functions in [`../app/scoring.py`](../app/scoring.py) against the notebook, and reloads the saved model in a fresh
  Python process to confirm it gives identical scores.

**Step 11: a second model (done, `model_training_step_11.ipynb`).**

* Builds the label (a bridge above Poor that ends in Poor), shows the first model cannot answer this question, and chooses the
  training history on the 2024 validation year (all years without California's 2016 to 2020 transitions again).
* Corrects the scores with a slope and intercept fitted on the validation year, which matches the real rates much better than a
  fixed shift for a rare event (the 130 highest scoring bridges are predicted at 13.8% against 13.8% real).
* Saves `app/model/poor_condition_model.joblib` in the same format as the first model, with two extra settings (it scores only
  bridges rated above 4, and shows "Already Poor" for the rest), then runs the same checks: the helper functions match the
  notebook, and a fresh Python process gives identical scores from both models.

### Part B: current risk snapshot for the app (done)

`scripts/build_current_risk_snapshot.py` builds each bridge's most recent (2025) record, joined with 2025 weather,
so the app can score current risk without reading the raw inventory files.

* The snapshot was re-built with the corrected weather. It joins on the bridge ID and the state, and a missing 2025
  coordinate is filled from the bridge's earlier years, as in the training data (525 bridges lacked a usable coordinate,
  and 53 still do after filling).
* The snapshot also carries each bridge's road and location names, cleaned of quotes and extra spaces, so the app can search by
  name.
* The final model does not use spending, so scoring does not need spending features. The app can show each state's most
  recent spending per bridge (2024, labelled as such) as background only.
* All 139,079 bridges have every column the model needs, and scoring them gives about 10% High, 21% Medium, and 69% Low.

### Part C: Streamlit app (done)

Started with `streamlit run app/app.py`. A **Risk view** choice in the sidebar switches every page between the two views of
risk (the chance of any rating drop, and the chance of falling into Poor), and each view is backed by its own model.

* **Home:** a short description, the two views side by side with how each tier really performed in 2025, and a chart of
  how each state's bridges split across the tiers.
* **Bridge Lookup:** pick a state and search by structure number, road, or place. It shows both models' estimated chance and
  tier for the bridge (or "Already Poor"), its key facts including its current condition, a chart of the biggest reasons behind
  the score for the selected view, the state's 2024 repair spending per bridge as background, and a small map.
* **State Risk Map:** a state's bridges colored by tier for the selected view, with a tier filter, a limit on how many bridges
  to draw (all High and Medium tier bridges are always kept), a choice of map background (OpenStreetMap, CARTO, or plain, shared with the Bridge Lookup map), and a table of the 20 highest
  scoring bridges.
* The app explains what the scores mean. A rating drop is not a failure, High in the first view mostly means bridges rated 8 or
  9, Poor does not mean unsafe, and the app does not predict collapse or tell anyone to close a bridge.
* All pages score through `app/scoring.py` (`load_models`, `score_all`, `explain_bridge`) and load data through `app/utils.py`, so a
  bridge shows the same numbers on every page.

## What the spending experiment found

* **The EDA.** Across states, spending per bridge and deterioration move together (correlation about +0.6). Within
  each state over time, the link is weak and slightly negative (about -0.1). See Section 9 of the EDA.
* **The model.** Adding spending per bridge lowered the 2025 ROC AUC by 0.0057, and the whole 95% bootstrap range was
  below zero. In the training years, years when a state spent more than usual had slightly more deterioration, so the
  model learned that more spending means more risk. In 2024 that reversed: New York spent about $15,000 less per bridge
  than usual and its deterioration jumped by more than 5 percentage points.
* **Why it failed.** There are only 35 distinct spending values (7 states by 5 years), and the model already has a
  `STATE` feature. Whatever spending could add has to come from small year to year changes within a state.
* **Spending may follow condition, not cause it.** Michigan and New York spend the most per bridge and also have the
  highest deterioration rates. The likely reason is that states direct money where bridges are already in poor
  shape. A positive link between spending and deterioration in this data should not be read as spending making
  bridges worse.
* **Possible refinements.** The dollars are not adjusted for inflation. And only 2020 to 2024 spending has been
  collected. Adding the SF-12A workbooks for 2010 to 2019 would give 105 state year values instead of 35, which would
  make a fairer test of research question 4.

## How each step gets checked

* Every notebook runs from top to bottom without errors before it is handed over.
* Joins must not change the row count of the model dataset. The weather join includes the state in its key for this
  reason, and the notebooks assert it.
* The training history is chosen on a validation year, not on the 2025 test set.
* The saved model is reloaded in a fresh session, and its predictions are compared against the notebook's to
  confirm nothing changed when it was saved.
* In the app, the same bridge must get the same risk score on the Bridge Lookup page and on the State Risk Map. This was checked
  by running every page headlessly (43 checks across both views and all seven states, including a bridge that is already Poor, a
  bridge with no coordinates, a search with no matches, the sampling used for large states, and the tier filter), by comparing the
  map's hover values with the scored table, and by starting the real server and looking at screenshots of every page.

## What is left

* Refresh the two slide decks. Both were built on the 2020 to 2025 results, and neither shows the app.
* Decide whether to keep the older training notebooks (`data_assembly` through `step_8`), which read the original single
  dataset file and only run if you still have a local copy.
* Collect the SF-12A spending workbooks for 2010 to 2019, which would give 105 state year values and a fairer test of research
  question 4.
* When the 2026 inventory is published, re-run `scripts/build_current_risk_snapshot.py` (changing the snapshot year) and the
  weather script, and retrain.
* The saved models were built with scikit-learn 1.9.1. If the app cannot load them on another machine, re-run
  `model_training_step_11.ipynb` there.
