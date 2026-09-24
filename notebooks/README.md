# Notebooks

## Exploratory data analysis

**`eda.ipynb`** covers the bridge inventory and weather data. It checks that all 42 raw bridge files share the same
columns, looks for data quality problems (duplicates, missing values, badly formatted coordinates), explores the
features and the target, builds the model ready dataset (`../data/processed/bridge_deterioration_dataset.csv.gz`),
and explores the weather data and how it joins to the bridges. The state bridge spending data has not been added
to this notebook yet.

## Model training

The model training notebook is being built one step at a time so the progress is easy to follow. Each file below
is a snapshot that contains every step up to and including its own, so the newest file is always the complete
version.

| Notebook | Adds |
|---|---|
| `model_training_data_assembly.ipynb` | Step 1: loads the bridge and weather data from GitHub (with a local fallback) and joins them |
| `model_training_step2.ipynb` | Step 2: defines the 10 categorical and 16 numeric features and cleans them |
| `model_training_step3.ipynb` | Step 3: time based split, training on transitions ending 2021 to 2024 and testing on 2025 |
| `model_training_step_4.ipynb` | Step 4: preprocessing and Logistic Regression pipeline, fit on the training set |

## Running the notebooks

The notebooks need `pandas`, `numpy`, `scikit-learn`, and `plotly`. If Jupyter reports a missing module even after
installing it, the install probably went to a different Python than the one your notebook uses. Running this in a
notebook cell installs into the right one:

```python
import sys
!{sys.executable} -m pip install pandas numpy scikit-learn plotly joblib
```
