"""Bridge Lookup: pick a state and a bridge, see its estimated risk, its key facts, and the reasons behind the score."""

import pandas as pd
import plotly.express as px
import streamlit as st

import scoring
import utils

view = utils.view_selector()
map_style = utils.background_selector()
models = utils.get_models()
bridges = utils.get_scored_bridges()

st.title("Bridge Lookup")
st.write("Pick a state, then search for a bridge by its structure number, the road it carries, or where it is.")

# ---------------------------------------------------------------- find a bridge
col_state, col_search = st.columns([1, 2])
state_code = col_state.selectbox("State", list(utils.STATE_NAMES), format_func=lambda code: utils.STATE_NAMES[code],
                                 index=list(utils.STATE_NAMES).index("MD"))
search = col_search.text_input("Search", placeholder="For example: 1CA0070, Interstate 5, or a creek name")

in_state = bridges[bridges["STATE"] == state_code]
if search.strip():
    text = (in_state[utils.KEY] + " " + in_state["FACILITY_CARRIED_007"] + " " + in_state["LOCATION_009"]).str.lower()
    matches = in_state[text.str.contains(search.strip().lower(), regex=False)]
else:
    matches = in_state

MAX_CHOICES = 200
if matches.empty:
    st.warning("No bridges match that search.")
    st.stop()

shown = matches.head(MAX_CHOICES)
labels = (shown[utils.KEY] + "  |  " + shown["FACILITY_CARRIED_007"].str.slice(0, 40) + "  |  "
          + shown["LOCATION_009"].str.slice(0, 40))
choice = st.selectbox(f"Bridge ({len(matches):,} match{'es' if len(matches) != 1 else ''}"
                      f"{f', showing the first {MAX_CHOICES}' if len(matches) > MAX_CHOICES else ''})",
                      list(shown.index), format_func=lambda idx: labels[idx])
bridge = bridges.loc[choice]

# ---------------------------------------------------------------- headline
st.divider()
road = bridge["FACILITY_CARRIED_007"] or "Road not listed"
place = bridge["LOCATION_009"] or "Location not listed"
st.subheader(f"{road}")
st.caption(f"{place}  |  Structure number {bridge[utils.KEY]}  |  {utils.STATE_NAMES[bridge['STATE']]}")

cards = st.columns(2)
for card, key in zip(cards, ("drop", "poor")):
    artifact = models[key]
    chance, tier = bridge[f"{key}_risk"], bridge[f"{key}_tier"]
    with card:
        with st.container(border=True):
            st.markdown(f"**{utils.VIEWS[key]['radio']}**")
            if tier == "Already Poor":
                st.metric("Estimated chance", "Already Poor")
                st.caption("This bridge's lowest rating is already 4 or below, so it cannot fall into Poor.")
            else:
                st.metric("Estimated chance", utils.format_chance(chance))
                st.markdown(f"Risk tier: **:{ {'High': 'red', 'Medium': 'orange', 'Low': 'blue'}[tier] }[{tier}]**")
                st.caption(utils.tier_sentence(artifact, tier))

if view == "drop" and bridge["drop_tier"] == "High":
    st.info(utils.VIEWS["drop"]["high_means"])

# ---------------------------------------------------------------- key facts
st.subheader("Key facts")
rating = bridge["LOWEST_RATING"]
material = utils.MATERIAL_NAMES.get(str(bridge["STRUCTURE_KIND_043A"]), "Unknown")
owner = utils.OWNER_NAMES.get(str(bridge["OWNER_022"]), f"code {bridge['OWNER_022']}")
facts = [
    ("Current condition", f"{utils.CONDITION_NAMES.get(bridge['BRIDGE_CONDITION'], 'Unknown')} (lowest rating {int(rating)} on a 0 to 9 scale)" if pd.notna(rating) else "unknown"),
    ("Age", f"{bridge['age']:.0f} years" if pd.notna(bridge["age"]) else "unknown"),
    ("Material", material),
    ("Owner", owner),
    ("Daily traffic", f"{bridge['ADT_029']:,.0f} vehicles" if pd.notna(bridge["ADT_029"]) else "unknown"),
    ("Truck share of traffic", f"{bridge['PERCENT_ADT_TRUCK_109']:.0f}%" if pd.notna(bridge["PERCENT_ADT_TRUCK_109"]) else "not reported"),
    ("Length", f"{bridge['STRUCTURE_LEN_MT_049']:,.0f} m" if pd.notna(bridge["STRUCTURE_LEN_MT_049"]) else "unknown"),
    ("Months between inspections", f"{bridge['INSPECT_FREQ_MONTHS_091']:.0f}" if pd.notna(bridge["INSPECT_FREQ_MONTHS_091"]) else "unknown"),
    ("Average temperature (2025)", f"{bridge['tavg_mean_c']:.1f} C" if pd.notna(bridge["tavg_mean_c"]) else "not available"),
    ("Freeze and thaw days (2025)", f"{bridge['freeze_thaw_days']:.0f}" if pd.notna(bridge["freeze_thaw_days"]) else "not available"),
]
facts_left, facts_right = st.columns(2)
for column, part in zip((facts_left, facts_right), (facts[:5], facts[5:])):
    column.markdown("\n".join(f"- **{name}:** {value}" for name, value in part))

# ---------------------------------------------------------------- why this score
st.subheader(f"Why this score: {utils.VIEWS[view]['lower']}")
artifact = models[view]
if bridge[f"{view}_tier"] == "Already Poor":
    st.write("This view only scores bridges that are not yet Poor. Switch the risk view in the sidebar to see the "
             "reasons for this bridge's chance of a rating drop.")
else:
    reasons = scoring.explain_bridge(bridges.loc[[choice]], artifact, top_n=8)
    reasons["Reason"] = reasons["feature"].map(utils.friendly_feature)
    fig = px.bar(reasons.iloc[::-1], x="contribution", y="Reason", orientation="h", color="effect",
                 color_discrete_map={"raises risk": "#c2362b", "lowers risk": "#5b8db8"},
                 labels={"contribution": "Effect on the score (log odds)", "Reason": "", "effect": ""})
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig)
    st.caption("Each bar is how much one fact about the bridge pushes its score up (red) or down (blue). Several facts can "
               "carry the same information, such as the lowest rating and the component ratings it comes from, so read the "
               "bars together rather than one at a time.")

# ---------------------------------------------------------------- context: spending and location
per_bridge, spending_year = utils.get_spending_context()
st.subheader("Background")
st.write(f"{utils.STATE_NAMES[state_code]} spent about **${per_bridge[state_code]:,.1f} thousand per bridge** on bridge "
         f"replacement and repair in {spending_year}. This is background only: state spending was tested as a predictor "
         "and made the predictions worse, so it is not used in either score.")

if pd.notna(bridge["lat"]) and pd.notna(bridge["lon"]):
    where = pd.DataFrame({"lat": [bridge["lat"]], "lon": [bridge["lon"]], "Bridge": [bridge[utils.KEY]]})
    fig = utils.scatter_map(where, map_style, lat="lat", lon="lon", hover_name="Bridge", zoom=12, height=380,
                            center={"lat": float(bridge["lat"]), "lon": float(bridge["lon"])})
    fig.update_traces(marker=dict(size=14, color="#c2362b"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)
else:
    st.caption("This bridge has no usable coordinates in the inspection record, so it cannot be shown on a map.")
