"""Shared loaders and display helpers for the BridgeWatch AI Streamlit app.

Every page gets its data and scores from here, so a bridge always shows the same numbers on every page.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import scoring

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
MODEL_DIR = APP_DIR / "model"
SNAPSHOT_PATH = REPO_ROOT / "data" / "processed" / "bridge_current_risk_input.csv.gz"
SPENDING_PATH = REPO_ROOT / "data" / "sf12a_state_bridge_spending_2020_2024.csv"
BRIDGE_COUNT_PATH = REPO_ROOT / "data" / "processed" / "bridge_deterioration_dataset_2024.csv.gz"

KEY = "STRUCTURE_NUMBER_008"
STATE_NAMES = {"CA": "California", "FL": "Florida", "MD": "Maryland", "MI": "Michigan", "NY": "New York",
               "TX": "Texas", "WA": "Washington"}

# Map backgrounds. Plotly draws the first two from tiles on the internet: OpenStreetMap uses plain image tiles from one
# server, and CARTO uses a vector style that needs several CARTO servers. "Plain" needs no internet at all.
BACKGROUNDS = {
    "Street map (OpenStreetMap)": "open-street-map",
    "Street map (CARTO)": "carto-positron",
    "Plain": "white-bg",
}

TIER_ORDER = ["High", "Medium", "Low"]
TIER_COLORS = {"High": "#c2362b", "Medium": "#f0a830", "Low": "#5b8db8", "Already Poor": "#7f7f7f"}

# The two views of risk. Each one is backed by its own trained model.
VIEWS = {
    "drop": {
        "radio": "Chance of any rating drop",
        "lower": "chance of any rating drop",
        "short": "rating drop",
        "question": "Will this bridge's condition rating drop by its next inspection?",
        "event": "had its lowest condition rating drop",
        "high_means": "A High tier bridge is likely to see its rating drop. That does not mean it is in bad shape: the "
                      "bridges most likely to drop are the ones rated 8 or 9, because they have the furthest to fall.",
    },
    "poor": {
        "radio": "Chance of falling into Poor condition",
        "lower": "chance of falling into Poor condition",
        "short": "falling into Poor",
        "question": "Will this bridge fall into Poor condition by its next inspection?",
        "event": "fell into Poor condition",
        "high_means": "A High tier bridge is likely to fall into Poor condition (a lowest rating of 4 or below). Bridges that "
                      "are already Poor are not scored.",
    },
}

RATING_NAMES = {0: "Failed", 1: "Imminent failure", 2: "Critical", 3: "Serious", 4: "Poor", 5: "Fair",
                6: "Satisfactory", 7: "Good", 8: "Very good", 9: "Excellent"}
CONDITION_NAMES = {"G": "Good", "F": "Fair", "P": "Poor"}
MATERIAL_NAMES = {"1": "Concrete", "2": "Concrete (continuous)", "3": "Steel", "4": "Steel (continuous)",
                  "5": "Prestressed concrete", "6": "Prestressed concrete (continuous)", "7": "Wood or timber",
                  "8": "Masonry", "9": "Aluminum or iron", "0": "Other"}
OWNER_NAMES = {"01": "State highway agency", "02": "County highway agency", "03": "Town or township highway agency",
               "04": "City or municipal highway agency", "11": "State park, forest, or reservation agency",
               "12": "Local park, forest, or reservation agency", "21": "Other state agency", "25": "Other local agency",
               "26": "Private (other than railroad)", "27": "Railroad", "31": "State toll authority",
               "32": "Local toll authority", "60": "Other federal agency", "61": "Indian tribal government",
               "62": "Bureau of Indian Affairs", "63": "Bureau of Fish and Wildlife", "64": "U.S. Forest Service",
               "66": "National Park Service", "68": "Bureau of Land Management", "69": "Bureau of Reclamation",
               "70": "Corps of Engineers (civil)"}
FEATURE_NAMES = {
    "LOWEST_RATING": "Lowest condition rating", "DECK_COND_058": "Deck rating",
    "SUPERSTRUCTURE_COND_059": "Superstructure rating", "SUBSTRUCTURE_COND_060": "Substructure rating",
    "CULVERT_COND_062": "Culvert rating", "SCOUR_CRITICAL_113": "Scour critical rating", "age": "Age (years)",
    "ADT_029": "Daily traffic", "PERCENT_ADT_TRUCK_109": "Truck share of traffic (%)",
    "MAIN_UNIT_SPANS_045": "Spans in main unit", "STRUCTURE_LEN_MT_049": "Length (m)",
    "DECK_WIDTH_MT_052": "Deck width (m)", "TRAFFIC_LANES_ON_028A": "Traffic lanes",
    "INSPECT_FREQ_MONTHS_091": "Months between inspections", "STRUCTURE_KIND_043A": "Material",
    "STRUCTURE_TYPE_043B": "Structure type", "DECK_STRUCTURE_TYPE_107": "Deck type", "OWNER_022": "Owner",
    "STATE": "State", "lat": "Latitude", "lon": "Longitude", "tmax_mean_c": "Average daily high (C)",
    "tmin_mean_c": "Average daily low (C)", "tavg_mean_c": "Average temperature (C)",
    "prcp_total_mm": "Yearly precipitation (mm)", "freeze_thaw_days": "Freeze and thaw days",
}


@st.cache_resource(show_spinner="Loading the models")
def get_models():
    """Both trained models, loaded once."""
    return scoring.load_models(MODEL_DIR)


@st.cache_data(show_spinner="Scoring every bridge")
def get_scored_bridges():
    """The 2025 snapshot with both models' chances and tiers added. One row per bridge."""
    models = get_models()
    text_cols = models["drop"]["categorical_cols"] + [KEY, "FACILITY_CARRIED_007", "LOCATION_009"]
    snapshot = pd.read_csv(SNAPSHOT_PATH, compression="gzip", dtype={col: str for col in text_cols})
    for col in ["FACILITY_CARRIED_007", "LOCATION_009"]:
        snapshot[col] = snapshot[col].fillna("") if col in snapshot.columns else ""
    scored = snapshot.join(scoring.score_all(snapshot, models))
    scored["state_name"] = scored["STATE"].map(STATE_NAMES)
    return scored


@st.cache_data
def get_spending_context():
    """Each state's most recent bridge repair spending per bridge, in thousands of dollars, and its year."""
    spending = pd.read_csv(SPENDING_PATH)
    year = int(spending["year"].max())
    latest = spending[spending["year"] == year].set_index("state")["bridge_repair_total_thousands"]
    counts = pd.read_csv(BRIDGE_COUNT_PATH, usecols=["STATE"])["STATE"].value_counts()
    return (latest / counts.reindex(latest.index)), year


def keep_choices():
    """Keep the sidebar choices when moving between pages (Streamlit forgets a widget's value on a page that does not show it).

    It has to run before any widget is created on the page, so view_selector() calls it first.
    """
    for key in ("view", "background"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def background_selector():
    """The sidebar choice of map background. Returns the plotly map style. Call it after view_selector()."""
    if "background" not in st.session_state:
        st.session_state["background"] = "Street map (OpenStreetMap)"
    st.sidebar.radio("Map background", list(BACKGROUNDS), key="background",
                     help="The street maps are downloaded from the internet. Choose Plain to draw only the bridges, for "
                          "example when working offline.")
    st.sidebar.caption("If the map behind the bridges stays blank, your network may be blocking that map provider. Try the "
                       "other street map, or choose Plain.")
    return BACKGROUNDS[st.session_state["background"]]


def scatter_map(df, style, **kwargs):
    """A plotly map of points. plotly 5.24 and later call it scatter_map; older versions call it scatter_mapbox."""
    make_map, style_argument = getattr(px, "scatter_map", None), "map_style"
    if make_map is None:
        make_map, style_argument = px.scatter_mapbox, "mapbox_style"
    return make_map(df, **{style_argument: style}, **kwargs)


def view_selector():
    """The sidebar choice between the two views of risk. The choice is kept when moving between pages."""
    keep_choices()
    if "view" not in st.session_state:
        st.session_state["view"] = "drop"
    st.sidebar.radio("Risk view", list(VIEWS), format_func=lambda key: VIEWS[key]["radio"], key="view",
                     help="Each view is its own model. The first asks if a bridge's rating will drop. The second asks "
                          "if a bridge that is not Poor now will fall into Poor.")
    st.sidebar.caption(VIEWS[st.session_state["view"]]["question"])
    return st.session_state["view"]


def format_chance(value):
    """A chance as text: 12.9%, under 0.1%, or a dash when there is none."""
    if pd.isna(value):
        return "n/a"
    if value < 0.001:
        return "under 0.1%"
    return f"{value:.1%}"


def format_rate(rate):
    """A real rate as text. Small rates get two decimals so 0.05% does not show as 0.0%."""
    return f"{rate:.2%}" if rate < 0.001 else f"{rate:.1%}"


def rating_text(rating):
    """9 becomes '9 (Excellent)'."""
    if pd.isna(rating):
        return "unknown"
    return f"{int(rating)} ({RATING_NAMES.get(int(rating), '')})"


def friendly_feature(feature):
    """Turn 'DECK_COND_058 = 7' into 'Deck rating: 7' and 'age = 13' into 'Age (years): 13'."""
    name, _, value = feature.partition(" = ")
    label = FEATURE_NAMES.get(name, name)
    if not value:
        return label
    if name == "STRUCTURE_KIND_043A":
        value = MATERIAL_NAMES.get(value, value)
    elif name == "OWNER_022":
        value = OWNER_NAMES.get(value, f"code {value}")
    elif value == "N":
        value = "not applicable"
    elif value == "nan":
        value = "missing"
    return f"{label}: {value}"


def tier_sentence(artifact, tier):
    """What happened to bridges in this tier on the 2025 test set, as a sentence."""
    stats = artifact["tier_summary"][tier]
    event = VIEWS[artifact["name"]]["event"]
    return (f"In the 2025 test year, {format_rate(stats['actual_rate'])} of {tier} tier bridges {event}, and the tier held "
            f"{stats['share_of_events']:.0%} of all the bridges that did.")
