"""State Risk Map: a state's bridges on a map, colored by risk tier."""

import numpy as np
import pandas as pd
import streamlit as st

import utils

view = utils.view_selector()
map_style = utils.background_selector()
bridges = utils.get_scored_bridges()
tier_col, risk_col = f"{view}_tier", f"{view}_risk"

st.title("State Risk Map")
st.write(f"Every bridge in a state, colored by its risk tier: {utils.VIEWS[view]['lower']}.")

col_state, col_tiers, col_limit = st.columns([1, 2, 1])
state_code = col_state.selectbox("State", list(utils.STATE_NAMES), format_func=lambda code: utils.STATE_NAMES[code],
                                 index=list(utils.STATE_NAMES).index("MD"))
tier_choices = utils.TIER_ORDER + (["Already Poor"] if view == "poor" else [])
tiers = col_tiers.multiselect("Tiers to show", tier_choices, default=utils.TIER_ORDER)
max_points = col_limit.number_input("Most bridges to draw", min_value=1000, max_value=60000, value=15000, step=1000,
                                    help="Large states have tens of thousands of bridges. When there are more than this, "
                                         "all High and Medium tier bridges are kept and a sample of the rest is drawn.")

in_state = bridges[bridges["STATE"] == state_code]
counts = in_state[tier_col].value_counts()
st.caption(" | ".join(f"{tier}: {counts.get(tier, 0):,}" for tier in tier_choices) + f" | All bridges: {len(in_state):,}")

shown = in_state[in_state[tier_col].isin(tiers)]
no_coordinates = shown["lat"].isna() | shown["lon"].isna()
if no_coordinates.any():
    st.caption(f"{int(no_coordinates.sum()):,} of these bridges have no usable coordinates and are not drawn.")
shown = shown[~no_coordinates]

if len(shown) > max_points:
    keep = shown[shown[tier_col].isin(["High", "Medium"])]
    rest = shown[~shown[tier_col].isin(["High", "Medium"])]
    room = max(int(max_points) - len(keep), 0)
    shown = pd.concat([keep, rest.sample(min(room, len(rest)), random_state=1)])
    st.caption(f"Drawing {len(shown):,} of the selected bridges. All High and Medium tier bridges are kept, and the rest are a "
               "random sample. Raise the limit above to draw more.")

if shown.empty:
    st.warning("There are no bridges to draw with these choices.")
    st.stop()

shown = shown.copy()
shown["Chance"] = shown[risk_col].map(utils.format_chance)
shown["Condition"] = shown["BRIDGE_CONDITION"].map(utils.CONDITION_NAMES).fillna("Unknown")
shown["Road"] = shown["FACILITY_CARRIED_007"].replace("", "not listed")
shown["Tier"] = shown[tier_col]

# Center the map on the bridges and pick a zoom that fits them (ignoring a few stray coordinates)
lat_low, lat_high = shown["lat"].quantile([0.005, 0.995])
lon_low, lon_high = shown["lon"].quantile([0.005, 0.995])
span = max(lat_high - lat_low, (lon_high - lon_low) * np.cos(np.radians(shown["lat"].median())), 0.05)
zoom = float(np.clip(np.log2(700 / span), 3, 11))
center = {"lat": float(shown["lat"].median()), "lon": float(shown["lon"].median())}

fig = utils.scatter_map(
    shown, map_style, lat="lat", lon="lon", color="Tier", color_discrete_map=utils.TIER_COLORS,
    category_orders={"Tier": tier_choices},
    hover_name=utils.KEY, hover_data={"Road": True, "Condition": True, "Chance": True, "lat": False, "lon": False, "Tier": False},
    zoom=zoom, center=center, height=650,
)
fig.update_traces(marker=dict(size=6, opacity=0.75))
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0, title_text=""))
st.plotly_chart(fig)

st.subheader(f"Highest risk bridges in {utils.STATE_NAMES[state_code]}")
top = (in_state[in_state[tier_col] != "Already Poor"].sort_values(risk_col, ascending=False).head(20)
       [[utils.KEY, "FACILITY_CARRIED_007", "LOCATION_009", "BRIDGE_CONDITION", "LOWEST_RATING", "age", risk_col, tier_col]]
       .rename(columns={utils.KEY: "Structure number", "FACILITY_CARRIED_007": "Road", "LOCATION_009": "Location",
                        "BRIDGE_CONDITION": "Condition", "LOWEST_RATING": "Lowest rating", "age": "Age", risk_col: "Chance",
                        tier_col: "Tier"}))
top["Condition"] = top["Condition"].map(utils.CONDITION_NAMES)
top["Chance"] = top["Chance"].map(utils.format_chance)
top["Age"] = top["Age"].round(0)
st.dataframe(top, hide_index=True)
st.caption("Look any of these up on the Bridge Lookup page to see the reasons behind the score. " + utils.VIEWS[view]["high_means"])
