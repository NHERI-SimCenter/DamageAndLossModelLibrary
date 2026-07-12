"""
1_About.py
----------
The About page for the DLML Explorer — a primarily-text page modeled on a good
project README.

It explains what the library and this web interface are, highlights the key
features as short user stories, and covers extension beyond FEMA P-58, how to
contribute, the license, acknowledgments, contact, and credits.

Section structure follows the directive from Adam Zsarnoczay.
"""

import streamlit as st

# set_page_config must be the first Streamlit command of the run, before the
# other imports (which may render widgets) execute.
st.set_page_config(
    page_title="About • DLML Explorer",
    page_icon="📊",
    layout="wide",
)

from dlml.web.st_ui.branding import render_contributors, render_header
from dlml.web.st_ui.sidebar import render_sidebar
from dlml.web.st_ui.theme import apply_theme


# ---------------- Header & sidebar ----------------
apply_theme()
render_sidebar()
render_header("DLML Explorer", subtitle="Damage and Loss Model Library")
st.caption("Open source · BSD 3-Clause License")

# ---------------- What is the DLML Explorer? ----------------
st.header("What is the DLML Explorer?")
st.markdown(
    """
**DLML Explorer** is a project from the NHERI
SimCenter that addresses a critical gap in natural hazards engineering: the
lack of a centralized, standardized, and easy-to-use repository for damage and
loss models. It provides the essential data — model parameters, descriptive
metadata, and configuration files — that power natural hazard risk assessment
simulations.

The **DLML Explorer** is the web interface to that library. It makes it easy to
browse the data, compare models side by side, and extract exactly the component
data you need for a simulation — without digging through the underlying CSV and
JSON files by hand.
    """
)

# ---------------- Key features ----------------
st.header("Key Features")
st.markdown(
    "Two user stories capture how most people use the Explorer — first to "
    "**discover** the right models, then to **collect and download** them for "
    "a project."
)

feat_search, feat_select = st.columns(2, gap="large")
with feat_search:
    st.subheader("1. Discover models")
    st.markdown(
        """
        Use the **search feature** and the **browse tree** to find the models
        you care about, then **compare fragility and consequence models** across
        different components to choose what fits your project.
        """
    )
with feat_select:
    st.subheader("2. Get data for your project")
    st.markdown(
        """
        As you browse, **add components to your selection**. When you're ready,
        **download them in one bundle** — formatted and ready to drop straight
        into your simulation.
        """
    )

# ---------------- Beyond FEMA P-58 ----------------
st.header("More Than FEMA P-58")
st.markdown(
    """
The library is **not limited to FEMA P-58**. It is built to grow with the
community: for example, the **NIST NED nonstructural fragilities** are expected
to be added by **mid-July 2026** as one such extension.

You can also **run the DLML Explorer locally** and point it at your own
**private fragilities** — adding them to the database so you can conveniently
display, discover, and select from them alongside the public models.
    """
)

with st.expander("Running locally"):
    st.markdown(
        """
**Prerequisites:** [Python 3.10 or newer](https://www.python.org/downloads/)
(3.12 recommended) and [Git](https://git-scm.com/downloads).
        """
    )

    st.markdown("**1. Clone the repository**")
    st.code(
        "git clone https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary.git\n"
        "cd DamageAndLossModelLibrary",
        language="bash",
    )

    st.markdown(
        "**2. Create and activate a virtual environment** (recommended, so the "
        "dependencies stay isolated from your system Python)"
    )
    st.code(
        "# macOS / Linux\n"
        "python3.12 -m venv .venv\n"
        "source .venv/bin/activate",
        language="bash",
    )
    st.code(
        "# Windows (PowerShell)\n"
        "py -3.12 -m venv .venv\n"
        ".venv\\Scripts\\Activate.ps1",
        language="powershell",
    )

    st.markdown("**3. Install the Explorer and its dependencies**")
    st.code('pip install ".[explorer]"', language="bash")

    st.markdown("**4. Launch the app**")
    st.code("dlml explorer", language="bash")

    st.markdown(
        "The Explorer opens in your browser at the URL Streamlit prints "
        "(typically [http://localhost:8501](http://localhost:8501)). The first "
        "time you search, the Explorer downloads a small embedding model "
        "(~64 MB) from HuggingFace and caches it — it fetches this up front at "
        "startup, so it happens only once. Press `Ctrl+C` in the terminal to "
        "stop.\n\n"
        "Once `dlml` is on PyPI you can skip the clone and install it directly "
        "with `pip install dlml[explorer]`."
    )

# ---------------- Contribute ----------------
st.header("Contribute")
st.markdown(
    """
Have fragility, consequence, or recovery models you'd like to see here? We'd
love your help building a growing library of **vetted and trusted** models.

We don't yet have a formal path for quality control and vetting, so for now
please **reach out (see Contact below) and join the conversation** on how to
build this out together. Our near-term plan is to collect new contributions in
a dedicated dataset (for example, **“FEMA P-58 ext”**), with clear references
to the contributors.

The NIST NED database is a valuable resource, but note that it is limited to
nonstructural fragilities supported by experimental data — a dedicated
extension dataset lets the community add high-quality models more broadly.
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
- **Adam Zsarnoczay** — NHERI SimCenter, Stanford University
    """
)

# ---------------- Contributor logos ----------------
render_contributors()
