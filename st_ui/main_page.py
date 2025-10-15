import streamlit as st
from auth.login_ui import render_login_panel
from st_search.fuzzy_visuals import render_fuzzy_search

st.set_page_config(
    page_title="Damage and Loss Model Library",
    page_icon="📊",
    layout="wide"
)
def render_main_page() -> None:
    """
    Render the main page
    """
    render_fuzzy_search()

