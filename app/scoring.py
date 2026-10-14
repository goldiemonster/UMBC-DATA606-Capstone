"""Scoring helpers shared by the model training notebook and the Streamlit app.

The trained model is saved by notebooks/model_training_step_10.ipynb as one joblib file,
app/model/deterioration_model.joblib. The file holds a dict (called the artifact below) with these keys:

    pipeline           fitted scikit-learn Pipeline: preprocessing plus Logistic Regression
    numeric_cols       numeric feature names
    categorical_cols   categorical feature names
    feature_cols       numeric_cols + categorical_cols, the columns the pipeline expects
    probability_shift  amount added to the model's log odds to turn its raw scores into real probabilities
    tier_cutoffs       {"medium": p, "high": p}, the chance cutoffs for the Medium and High risk tiers
    test_metrics       the model's scores on the 2025 test set
    coef_table         list of dicts with feature, coefficient, and odds_multiplier
    training           a description of the training data
    versions           the library versions used to train the model

The functions below only need that dict, so the app never depends on code from the notebook.
"""

import joblib
import numpy as np
import pandas as pd

TIERS = ["Low", "Medium", "High"]


def load_model(path):
    """Load the saved model artifact."""
    return joblib.load(path)


def prepare_features(df, artifact):
    """Pick the model's feature columns and clean them exactly the way the training notebook did.

    Categorical columns are turned into text, with blank strings treated as missing. Numeric columns are
    converted to numbers. A snapshot read from csv should have its categorical columns read as text
    (dtype=str), so a code such as 01 does not turn into 1.
    """
    features = df[artifact["feature_cols"]].copy()
    for col in artifact["categorical_cols"]:
        features[col] = features[col].astype(str).replace("", np.nan)
    for col in artifact["numeric_cols"]:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    return features


def score_bridges(df, artifact):
    """Score every row of df. Returns a DataFrame (same index) with the estimated chance and the risk tier.

    The chance is the model's raw score with the probability shift applied, so it can be read as the estimated
    probability that the bridge's lowest condition rating drops by its next inspection. The tier is Low, Medium,
    or High, based on the cutoffs saved with the model.
    """
    log_odds = artifact["pipeline"].decision_function(prepare_features(df, artifact))
    chance = 1.0 / (1.0 + np.exp(-(log_odds + artifact["probability_shift"])))
    cutoffs = artifact["tier_cutoffs"]
    tier = np.where(chance >= cutoffs["high"], "High", np.where(chance >= cutoffs["medium"], "Medium", "Low"))
    return pd.DataFrame({"risk": chance, "tier": tier}, index=df.index)


def readable_name(encoded_name, categorical_cols):
    """Turn 'num__age' into 'age' and 'cat__DECK_COND_058_5' into 'DECK_COND_058 = 5'."""
    kind, feature = encoded_name.split("__", 1)
    if kind == "num":
        return feature
    for col in categorical_cols:
        if feature.startswith(col + "_"):
            return f"{col} = {feature[len(col) + 1:]}"
    return feature


def explain_bridge(row, artifact, top_n=6, decimals=3):
    """The biggest reasons behind one bridge's score. row is a one row DataFrame.

    Each feature's contribution is its coefficient times its (scaled or 0/1) value. The contributions plus the
    model's intercept add up to the bridge's raw log odds, so the largest contributions are the reasons for the score.
    The contributions are rounded to `decimals` places for display (use decimals=None for the exact values).
    """
    features = prepare_features(row, artifact)
    preprocess = artifact["pipeline"].named_steps["preprocess"]
    clf = artifact["pipeline"].named_steps["clf"]

    x = preprocess.transform(features)
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    contributions = x * clf.coef_[0]

    names = [readable_name(n, artifact["categorical_cols"]) for n in preprocess.get_feature_names_out()]
    parts = pd.DataFrame({"feature": names, "contribution": contributions})
    parts = parts[parts["contribution"] != 0]
    parts = parts.reindex(parts["contribution"].abs().sort_values(ascending=False).index).head(top_n)

    # Show numeric features with the bridge's actual value, for example "age = 13"
    for col in artifact["numeric_cols"]:
        parts.loc[parts["feature"] == col, "feature"] = f"{col} = {features[col].iloc[0]:g}"
    parts["effect"] = np.where(parts["contribution"] > 0, "raises risk", "lowers risk")
    if decimals is not None:
        parts = parts.round(decimals)
    return parts.reset_index(drop=True)
