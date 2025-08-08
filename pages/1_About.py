import streamlit as st
import pandas as pd
import altair as alt
from datetime import date

# ---------------- Page config & sidebar ----------------
st.set_page_config(
    page_title="About • Damage & Loss Library",
    page_icon="🏛️",
    layout="wide",
)

# Sidebar branding / quick nav
with st.sidebar:
    # st.image("assets/simcenter_logo.png", width=160)   # Place your logo file in assets/
    st.header("NHERI SimCenter")
    st.markdown(
        """
        **Resilient and Sustainable Infrastructure**  
        Department of Civil & Environmental Engineering  
        Stanford University
        """
    )
    st.markdown("---")
    st.page_link("https://simcenter.designsafe-ci.org", label="Official website 🌐")

# ---------------- Hero section ----------------
st.title("About")
st.caption(
    "Interactive web app built with **Streamlit** to explore, compare and download "
    "fragility, consequence, and recovery models curated by the NHERI SimCenter."
)
st.success("💡 This page is a live demo—poke around!")

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Models", "82", "▲ 9 new")
col2.metric("Materials", "14", delta="—")
col3.metric("Last update", date.today().strftime("%b %d, %Y"))

# ---------------- Interactive model explorer ----------------
st.header("🔍 Quick Explorer")

hazards = ["Earthquake", "Hurricane", "Flood", "Wildfire"]
materials = ["Steel", "Concrete", "Timber", "Masonry"]

hazard = st.selectbox("Choose a hazard type:", hazards, index=0)
mat = st.selectbox("Choose primary material:", materials, index=0)
min_year, max_year = st.slider("Year published:", 1995, 2025, (2010, 2025))

# Fake data table (replace with real query)
sample = pd.DataFrame({
    "Model Name": [f"{hazard[:3]}-{mat[:2]}-{yr}" for yr in range(min_year, max_year, 3)],
    "Hazard": hazard,
    "Material": mat,
    "Year": list(range(min_year, max_year, 3)),
    "Authors": ["Doe et al."]*len(range(min_year, max_year, 3)),
})
st.dataframe(sample, use_container_width=True, hide_index=True)

# ---------------- Tiny viz demo ----------------
with st.expander("📊 Library growth over time (placeholder)"):
    # Mock data
    growth = pd.DataFrame({
        "Year": list(range(2010, 2026)),
        "Models": [5,7,10,14,18,24,30,37,46,54,62,70,78,82,82,82],
    })
    chart = (
        alt.Chart(growth)
        .mark_area(interpolate="monotone")
        .encode(x="Year:O", y="Models:Q", tooltip=["Year","Models"])
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------- Collapsible details ----------------
with st.expander("ℹ️ What is a Damage & Loss Model?"):
    st.markdown(
        """
        A **damage and loss model** links the intensity of a hazard to physical 
        damage states and quantifies either economic loss or downtime.  
        They are the backbone of modern **Performance‐Based Engineering** workflows,
        enabling decision-makers to balance **risk, cost, and resilience**.
        """
    )

with st.expander("🛠️ Built with Streamlit (why we love it)"):
    st.markdown(
        """
        * **Markdown + LaTeX** for rich text  
        * **Widgets** (`st.selectbox`, `st.slider`, `st.toggle`, …) for instant interactivity  
        * **DataFrames** with in-browser sorting & filtering  
        * **Altair / Plotly** one-liner charts  
        * **Session State** to remember user choices  
        * **st.download_button** for on-the-fly file export  
        * **st.experimental_data_editor** for quick what-ifs  
        * **st.secrets / caching** for fast DB calls  
        * **Theming**—brand the app with your colors & logo
        """
    )

# ---------------- Call-to-action ----------------
st.subheader("📥 Download a sample model")
st.download_button(
    label="Get JSON",
    data='{"model":"Sample"}',
    file_name="sample_model.json",
    mime="application/json",
)

# ---------------- Footnote ----------------
st.markdown("---")
st.markdown(
    "Made with ❤️ by the **NHERI SimCenter** research team at Stanford University. "
    "This project is supported by the **National Science Foundation (CMMI)**."
)
st.caption("Page generated with Streamlit 💫")
