# Notebooks

## Exploratory data analysis

**`eda.ipynb`** covers all three data sources. It checks that all 112 raw bridge files (7 states by 16 years) have
the columns the project needs, rebuilds the two condition summaries FHWA only published from 2019 on, looks for data
quality problems (duplicate structure numbers, missing values, badly formatted coordinates, bridge numbers that change
between years), explores the features (2025 snapshot) and checks that they mean the same thing in the older years (Section 4.6), explores the target, and builds the model ready dataset. The dataset is saved as 15
files, one per feature year, in `../data/processed/` (`bridge_deterioration_dataset_2010.csv.gz` through
`_2024.csv.gz`). Section 5 shows how the deterioration rate differs by state and year, which is where California's
unusual 2016 to 2020 ratings show up. Section 8 explores the weather data and Section 9 the state bridge spending
data, each ending with how it joins to the bridge dataset. Section 9 also compares spending per bridge with
deterioration rates, across states and within each state over time.

## Model training

The model training notebook was built one step at a time so the progress is easy to follow. Each file below is a
snapshot that contains every step up to and including its own, so the newest file is always the complete version.

| Notebook | Adds |
|---|---|
| `model_training_data_assembly.ipynb` | Step 1: loads the bridge and weather data from GitHub (with a local fallback) and joins them |
| `model_training_step2.ipynb` | Step 2: defines the 10 categorical and 16 numeric features and cleans them |
| `model_training_step3.ipynb` | Step 3: time based split, training on transitions ending 2021 to 2024 and testing on 2025 |
| `model_training_step_4.ipynb` | Adds the spending data, the preprocessing and Logistic Regression pipeline, and the evaluation on the 2025 test set (ROC AUC, precision and recall, and the capture rate) |
| `model_training_step_6.ipynb` | Compares the model with and without weather and with and without spending, including a bootstrap check on the differences |
| `model_training_step_7.ipynb` | Builds the final model (bridge and weather features, no spending) |
| `model_training_step_8.ipynb` | Interprets the final model: which features raise or lower risk, and how to explain a single bridge's score |
| `model_training_step_9.ipynb` | Extends everything to the full 2010 to 2025 history and chooses how much history to train on |
| `model_training_step_10.ipynb` | Saves the final model for the app: corrects the scores into real probabilities, defines the Low, Medium, and High risk tiers, writes `app/model/deterioration_model.joblib`, and checks that the saved file gives identical scores in a fresh Python process |
| `model_training_step_11.ipynb` | Builds and saves a second model for the question "will a bridge that is not Poor yet fall into Poor?", writes `app/model/poor_condition_model.joblib`, and scores today's bridges with both models |

The notebooks up to `model_training_step_8.ipynb` were built on the original 2020 to 2025 data, and they read the
original single file `bridge_deterioration_dataset.csv.gz`, which has since been replaced by the 15 yearly files. They
are kept to show how the work progressed, and they only run if you still have that single file locally.
**`model_training_step_11.ipynb` is the current, complete version and runs against the repository as it is now.** It is `model_training_step_10.ipynb` plus Section 10, which builds the second model.

What step 9 changes compared with step 8:

- It loads the 15 yearly bridge files and 16 yearly weather files. The weather join now uses the state as well as the
  bridge ID and year, because a structure number is only unique within a state.
- The spending experiments (Sections 3 to 6) use only the 2020 and later records, since spending exists from 2020.
  The conclusions are the same as before: spending made the model worse, and weather made no clear difference.
- Section 7 decides how much history the final model should train on. It compares three choices on a validation year
  (2024): all years, all years without California's 2016 to 2020 transitions, and 2020 on only. Training on all years
  with nothing removed was clearly worst, because California's ratings swung sharply in those years. Leaving them
  out matched the 2020 on model on the validation year while using almost four times as much data, so it was chosen.
- On the 2025 test set the final model scores a ROC AUC of 0.726, against 0.722 for the model trained on 2020 on. The
  bootstrap range for that gain is +0.0025 to +0.0062.
- Section 8 reads the coefficients of the new final model.

What step 10 adds:

- The raw model scores are far too high as probabilities (they average 44% when only 4.5% of bridges really dropped), because
  balanced class weights inflate them. Section 9 corrects them with one fixed shift in the log odds. The corrected chances
  track the real drop rates closely, and the ranking of bridges does not change.
- The High tier (the top 10% of bridges) holds 31% of the drops, and a High tier bridge dropped 14.1% of the time,
  against 2.5% in the Low tier.
- The helper functions the app calls are in [`../app/scoring.py`](../app/scoring.py), and this notebook tests them.

What step 11 adds:

- The first model's High tier is mostly bridges rated 8 or 9, and its ROC AUC for finding the bridges that fall into Poor is only
  0.55. Section 10 builds a second model for that question, with the same features, the same Logistic Regression, the same
  time based split, and a validation year to choose the training history (again all years without California's 2016 to 2020
  transitions).
- It is much better at its own question: a ROC AUC of 0.89, and its top 20% of bridges contain 84% of the bridges that fell into
  Poor in 2025 (the first model's top 20% contain 29%).
- Falling into Poor is rare (about 0.6% of bridges), so the fixed shift used for the first model overstates the highest scores.
  This model's scores are corrected with a slope and intercept fitted on the validation year, which makes the top of the ranking
  match what really happened.
- Bridges that are already Poor are not scored by this model. The app shows them as "Already Poor".
- The two views disagree a lot: only 0.7% of the bridges above Poor are High tier in both. The app offers both.

The notebooks are finished. The Streamlit app that uses the saved models is in [`../app/`](../app).

## Running the notebooks

The notebooks need `pandas`, `numpy`, `scikit-learn`, and `plotly`. If Jupyter reports a missing module even after
installing it, the install probably went to a different Python than the one your notebook uses. Running this in a
notebook cell installs into the right one:

```python
import sys
!{sys.executable} -m pip install pandas numpy scikit-learn plotly joblib
```

The notebooks load their data from this GitHub repository first and fall back to the local `../data` folder if the
download fails, so push any new data files before running them, or keep a local copy of the `data` folder next to
the notebooks folder.
