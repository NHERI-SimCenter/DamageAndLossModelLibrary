"""
1_Browse_and_Search.py
----------------------
The Browse & Search page — the interactive tool for finding, comparing, and
collecting damage and loss models. The About page (``app.py``) is the landing
page; this is reached from the top-bar navigation or the call-to-action there.
"""

import streamlit as st

# set_page_config must be the first Streamlit command of the run, before the
# sidebar (which renders widgets) executes.
st.set_page_config(
    page_title="Browse & Search • DLML Explorer",
    page_icon="📊",
    layout="wide",
)

from dlml.web.st_ui.theme import apply_theme
from dlml.web.st_ui.sidebar import render_sidebar
from dlml.web.st_ui.main_page import render_main_page

apply_theme()
render_sidebar()
render_main_page()
