"""The BridgeWatch AI home page. The app is started with app.py, which sets up the pages."""

import pandas as pd
import plotly.express as px
import streamlit as st

import utils

view = utils.view_selector()
models = utils.get_models()
bridges = utils.get_scored_bridges()

st.title("BridgeWatch AI")
st.write(
    "BridgeWatch AI estimates which bridges are most likely to get worse by their next inspection. It uses federal "
    "inspection records (the National Bridge Inventory) for seven states, 2010 to 2025, together with the weather at "
    "each bridge. Look up a single bridge on the **Bridge Lookup** page, or see a whole state on the **State Risk Map** page."
)

st.subheader("Two ways to look at risk")
st.write(
    "The app answers two different questions, and each one has its own model. Use the **Risk view** choice in the sidebar "
    "to switch between them on any page."
)
left, right = st.columns(2)
for column, key in zip((left, right), ("drop", "poor")):
    artifact = models[key]
    with column:
        st.markdown(f"**{utils.VIEWS[key]['radio']}**")
        st.write(utils.VIEWS[key]["question"])
        st.caption(utils.VIEWS[key]["high_means"])
        total = sum(s["bridges"] for s in artifact["tier_summary"].values())
        rows = []
        for tier in utils.TIER_ORDER:
            stats = artifact["tier_summary"][tier]
            rows.append({"Tier": tier, "Bridges": f"{stats['bridges'] / total:.0%}",
                         "Real rate": utils.format_rate(stats["actual_rate"]),
                         "Events found": f"{stats['share_of_events']:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"Bridges: share of the bridges in each tier. Real rate: share of the tier's bridges that "
                   f"{utils.VIEWS[key]['event']} in the 2025 test year. Events found: share of all the bridges that did "
                   "that sit in the tier.")
        metrics = artifact["test_metrics"]
        st.caption(f"Model accuracy on the 2025 test year: ROC AUC {metrics['ROC AUC']:.2f}. Its top 20% of bridges "
                   f"contain {metrics['Found in top 20%']:.0%} of the bridges that {utils.VIEWS[key]['event']}.")

st.subheader(f"Where the bridges stand: {utils.VIEWS[view]['lower']}")
tier_column = f"{view}_tier"
counts = pd.crosstab(bridges["state_name"], bridges[tier_column])
order = [tier for tier in utils.TIER_ORDER + ["Already Poor"] if tier in counts.columns]
shares = (counts[order].div(counts[order].sum(axis=1), axis=0) * 100).round(1).reset_index().melt(
    id_vars="state_name", var_name="Tier", value_name="Share of the state's bridges (%)")
fig = px.bar(shares, x="state_name", y="Share of the state's bridges (%)", color="Tier", color_discrete_map=utils.TIER_COLORS,
             category_orders={"Tier": order}, labels={"state_name": "State"})
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
st.plotly_chart(fig)
st.caption(f"{len(bridges):,} bridges scored in {bridges['STATE'].nunique()} states, using each bridge's 2025 inspection record. "
           "The tier cutoffs were set on the 2025 test bridges, so the share in each tier differs a lot by state. That comes "
           "from the model, not from a rule of the same share per state.")

with st.expander("How to read the scores"):
    st.markdown(
        "- **The chance** is the model's estimate that the event happens by the bridge's next inspection. It is an estimate, "
        "and the model is better at ranking bridges than at predicting any single one.\n"
        "- **The tier** groups bridges by that chance. High is the top 10% of bridges in the 2025 test year, Medium the next "
        "20%, and Low the rest.\n"
        "- **A rating drop is not a failure.** The first model counts any drop of one point or more, so a bridge going from 9 "
        "to 8 counts the same as one going from 4 to 3.\n"
        "- **Poor** is the federal term for a bridge whose lowest rating is 4 or below. Poor does not mean unsafe. It means "
        "the bridge needs attention.\n"
        "- The app does not predict collapse or tell anyone whether to close a bridge."
    )
