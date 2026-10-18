"""Scoring helpers shared by the model training notebook and the Streamlit app.

The app offers two views of risk, and each one has its own trained model saved in app/model/:

    "drop"  deterioration_model.joblib   the chance a bridge's lowest condition rating drops by its next inspection
    "poor"  poor_condition_model.joblib  the chance a bridge that is not Poor now falls into Poor by its next inspection

Each file holds a dict (called the artifact below) with these keys:

    name               "drop" or "poor"
    title              a short title for the view, for example "Chance of falling into Poor condition"
    pipeline           fitted scikit-learn Pipeline: preprocessing plus Logistic Regression
    numeric_cols       numeric feature names
    categorical_cols   categorical feature names
    feature_cols       numeric_cols + categorical_cols, the columns the pipeline expects
    probability_slope  multiplier for the model's log odds before the shift is added (1.0 when only a shift is needed)
    probability_shift  amount added to the model's log odds to turn its raw scores into real probabilities
    tier_cutoffs       {"medium": p, "high": p}, the chance cutoffs for the Medium and High risk tiers
    tier_summary       what happened to bridges in each tier on the 2025 test set
    eligible_min_rating  None, or the lowest LOWEST_RATING a bridge can have and still be scored
    ineligible_tier    the tier shown for bridges that are not scored (for example "Already Poor")
    test_metrics       the model's scores on the 2025 test set
    coef_table         list of dicts with feature, coefficient, and odds_multiplier
    training           a description of the training data
    versions           the library versions used to train the model

The functions below only need those dicts, so the app never depends on code from the notebooks.
"""

import os

import joblib
import numpy as np
import pandas as pd

TIERS = ["Low", "Medium", "High"]
MODEL_FILES = {"drop": "deterioration_model.joblib", "poor": "poor_condition_model.joblib"}


def load_model(path):
    """Load one saved model artifact."""
    return joblib.load(path)


def load_models(model_dir):
    """Load both saved models. Returns {"drop": artifact, "poor": artifact}."""
    return {name: load_model(os.path.join(model_dir, filename)) for name, filename in MODEL_FILES.items()}


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


def eligible_rows(df, artifact):
    """True for the bridges this model scores. The Poor model skips bridges that are already Poor."""
    min_rating = artifact.get("eligible_min_rating")
    if min_rating is None:
        return pd.Series(True, index=df.index)
    return pd.to_numeric(df["LOWEST_RATING"], errors="coerce") >= min_rating


def score_bridges(df, artifact):
    """Score every row of df. Returns a DataFrame (same index) with the estimated chance and the risk tier.

    The chance is the model's raw score with the probability slope and shift applied, so it can be read as the
    estimated probability of the event the model was trained on. The tier is Low, Medium, or High, based on the cutoffs
    saved with the model. Bridges the model does not score (see eligible_rows) get no chance and the model's
    ineligible tier, for example "Already Poor".
    """
    log_odds = artifact["pipeline"].decision_function(prepare_features(df, artifact))
    chance = 1.0 / (1.0 + np.exp(-(artifact.get("probability_slope", 1.0) * log_odds + artifact["probability_shift"])))
    cutoffs = artifact["tier_cutoffs"]
    tier = np.where(chance >= cutoffs["high"], "High", np.where(chance >= cutoffs["medium"], "Medium", "Low"))
    scores = pd.DataFrame({"risk": chance, "tier": tier}, index=df.index)

    skipped = ~eligible_rows(df, artifact)
    if skipped.any():
        scores.loc[skipped, "risk"] = np.nan
        scores.loc[skipped, "tier"] = artifact.get("ineligible_tier", "Not scored")
    return scores


def score_all(df, artifacts):
    """Score df with every model. Columns are named like drop_risk, drop_tier, poor_risk, poor_tier."""
    parts = []
    for name, artifact in artifacts.items():
        parts.append(score_bridges(df, artifact).rename(columns={"risk": f"{name}_risk", "tier": f"{name}_tier"}))
    return pd.concat(parts, axis=1)


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
