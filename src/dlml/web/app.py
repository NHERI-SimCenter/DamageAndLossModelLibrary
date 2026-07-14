"""
app.py
------
Entry point and landing page for the DLML Explorer — a primarily-text About
page modeled on a good project README. It explains what the library and this web
interface are, how it's run, and how to contribute, then sends people into the
Browse & Search tool (``pages/1_Browse_and_Search.py``) via a prominent call to
action and the top-bar navigation.
"""

import streamlit as st

# set_page_config must be the first Streamlit command of the run, before the
# other imports (which may render widgets) execute.
st.set_page_config(
    page_title="DLML Explorer",
    page_icon="📊",
    layout="wide",
)

from dlml.web.st_ui.branding import (
    render_contributors,
    render_header,
    render_top_bar,
)
from dlml.web.st_ui.sidebar import render_sidebar
from dlml.web.st_ui.theme import apply_theme


# ---------------- Header, nav & sidebar ----------------
apply_theme()
render_sidebar()
render_top_bar()
render_header("DLML Explorer", subtitle="Damage and Loss Model Library")
st.caption(
    "A growing, open library of standardized damage and loss models for "
    "natural-hazards engineering."
)

# ---------------- Call to action ----------------
# The primary way into the tool — visible up front so first-time visitors don't
# have to discover the top-bar link on their own.
if st.button(
    "🔍 Browse & Search the model library →",
    type="primary",
    key="cta_browse",
):
    st.switch_page("pages/1_Browse_and_Search.py")

# ---------------- What is the DLML Explorer? ----------------
st.header("What is the DLML Explorer?")
st.markdown(
    """
NHERI SimCenter's **[Damage and Loss Model Library (DLML)](https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary)** is a curated,
open-source, **version-controlled** library of damage and loss models — the 
fragility and consequence data that power natural-hazard risk simulations. It 
gathers models from established sources such as **FEMA P-58** and **Hazus**, 
converts them into a single standardized format, and pairs every model with 
**rich metadata and a reference to its source**, so any model can be traced 
back to its origin.

The **DLML Explorer** is the web interface to that library. It lets you browse
the data, compare models side by side, and pull exactly the component data your
project needs — without digging through raw CSV and JSON files by hand.

The vision is a **living, community-driven library** for practicing engineers
and researchers alike: one that features the several primary datasets in use 
today and keeps growing as the field advances.
    """
)

# ---------------- Using the Explorer ----------------
st.header("Using the Explorer")
st.markdown(
    "Most people use the Explorer to **discover** the right models and then "
    "**collect and download** them for a project."
)

st.subheader("1. Discover models")
st.markdown(
    "Use the **search** and the **browse tree** to find the models you care "
    "about, then **compare fragility and consequence models** across components "
    "— with interactive plots and full parameter tables — to choose what fits "
    "your project."
)

st.subheader("2. Get data for your project")
st.markdown(
    "As you browse, **add models to your selection**, then **download them in "
    "one bundle**. The data is directly compatible with "
    "[Pelicun](https://github.com/NHERI-SimCenter/pelicun) — the SimCenter's "
    "open-source damage-and-loss engine, which uses the DLML as its model "
    "source — so you can run with it right away, or hand the CSVs to any other "
    "tool."
)

# ---------------- Governance & privacy ----------------
st.header("Governance and privacy")
st.markdown(
    """
The DLML Explorer is **managed by the NHERI SimCenter**. Degenkolb Engineers
generously supported its development — but Degenkolb does not host or operate it.

- **Your work stays yours.** No one — not even the SimCenter — monitors which
  models you review or collects data on how you use the Explorer.
- **Metadata as published.** For large published datasets like FEMA P-58 and
  Hazus, the technical notes and metadata are reproduced **as-is from the
  original source** — the SimCenter does not edit, re-interpret, or add to them.
- **Traceable and reproducible.** Every model carries a reference to its source,
  and the whole library is version-controlled — so an analysis can be tied to an
  exact version of the data, for reproducibility and accountability.
    """
)

# ---------------- More than FEMA P-58 ----------------
st.header("More than FEMA P-58")
st.markdown(
    """
The library is **not limited to FEMA P-58**, and it is **hazard-agnostic** — it
already spans seismic and hurricane models. The **SimCenter Wind Component
Library** is a good template for what a purpose-built, well-referenced dataset
can look like, and a model for how the seismic side can grow.

It is also built to extend. Vulnerability models from the
**[NIST Nonstructural Element Database](https://github.com/usnistgov/NED)** will join the library soon as
one such addition. And because the Explorer can
**run locally against your own data**, you can add your **private fragilities**
right alongside the public models and use the Explorer to display, discover, 
and select from them.
    """
)

# Wrapped in a keyed container so the theme can give the expander a little more
# room above it (see the .st-key-ril-block rule) without loosening the tree.
with st.container(key="ril-block"):
    with st.expander("Run it locally — and add your own private models"):
        st.markdown(
            "All you need is "
            "[Python 3.10 or newer](https://www.python.org/downloads/) "
            "(3.12 recommended). Install the Explorer and launch it:"
        )
        st.code(
            "pip install simcenter-dlml[explorer]\ndlml explorer",
            language="bash",
        )
        st.markdown(
            "It opens in your browser at the address Streamlit prints — typically "
            "[http://localhost:8501](http://localhost:8501); press `Ctrl+C` in the "
            "terminal to stop. Running locally, you can point the Explorer at your "
            "own data and display, discover, and select from your private models "
            "right alongside the public ones."
        )

# ---------------- Contribute ----------------
st.header("Contribute")
st.markdown(
    """
Do you have fragility or consequence models you'd like to see here? We'd
love your help building a growing library of **vetted, well-referenced** models,
and **we're glad to help you get them in** — reach out (see Contact below) and
join the conversation. We don't have a formal submission-and-vetting process
yet, and **making contributing easier is on our near-term roadmap**; for now 
the plan is to collect new contributions in a dedicated dataset (for example,
**"FEMA P-58 ext"** for seismic component models), and give clear credit to the
contributors.

**What we ask:** every model needs **at least one supporting reference** — a
publication or report that documents it — just as the SimCenter Wind Component
Library does. This is to keep the library trustworthy.
    """
)

# ---------------- License ----------------
st.header("License")
st.markdown(
    """
The DLML Explorer and the underlying Damage and Loss Model Library are
distributed under the **BSD 3-Clause License**. You are free to use the data
for any purpose, **including commercial use**. See the `LICENSE` file for the
full terms.
    """
)

# ---------------- Acknowledgments ----------------
st.header("Acknowledgments")
st.markdown(
    """
We gratefully acknowledge the generous in-kind contribution from Degenkolb
Engineers, which supported the development work on the DLML Explorer. This
material is based upon work supported by the U.S. National Science Foundation
under Grants No. 1612843 and No. 2131111. Any opinions, findings, conclusions,
or recommendations expressed in this material are those of the author(s) and do
not necessarily reflect the views of the U.S. National Science Foundation.

We also wish to express our gratitude to colleagues at the NHERI SimCenter and
Degenkolb Engineers for their feedback and input. The insights shared by many
engineers and experts in the natural hazards engineering community were
instrumental in shaping the priorities for this work. In particular,
discussions with Dustin Cook (NIST), Curt Haselton (HBRisk), Jon Heintz
(Applied Technology Council), John Hooper (MKA), James Malley (Degenkolb),
Peter Morris (AECOM), and Robert Pekelnicky (Degenkolb) were especially
valuable. We also appreciate the collaboration with the NIST NED development
team.
    """
)

# ---------------- Contact ----------------
st.header("Contact")
st.markdown(
    "Adam Zsarnoczay — NHERI SimCenter, Stanford University — "
    "[adamzs@stanford.edu](mailto:adamzs@stanford.edu)"
)

# ---------------- Developed by ----------------
st.header("Developed by")
st.markdown(
    """
- **Tshajlij Lee** — Degenkolb Engineers
- **Hannah Thompson** — Degenkolb Engineers
- **Insung Kim** — Degenkolb Engineers
- **Adam Zsarnoczay** — NHERI SimCenter, Stanford University
    """
)

# ---------------- Contributor logos ----------------
render_contributors()
