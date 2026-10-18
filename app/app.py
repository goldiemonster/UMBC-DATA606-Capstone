"""BridgeWatch AI. Start the app with:  streamlit run app/app.py

This file sets up the page settings and the navigation. Each page is its own script in the screens folder.
"""

import streamlit as st

st.set_page_config(page_title="BridgeWatch AI", page_icon="\U0001F309", layout="wide")

pages = [
    st.Page("screens/home.py", title="Home", url_path="home", default=True),
    st.Page("screens/bridge_lookup.py", title="Bridge Lookup", url_path="bridge-lookup"),
    st.Page("screens/state_risk_map.py", title="State Risk Map", url_path="state-risk-map"),
]
st.navigation(pages).run()
