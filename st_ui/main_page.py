import streamlit as st
from st_search.fuzzy_visuals import render_fuzzy_search
from st_visuals.tree_visuals import render_seismic_tree, render_wind_tree


def render_main_page() -> None:
    """
    Render the main page
    """
    st.set_page_config(
        page_title="Damage and Loss Model Library",
        page_icon="📊",
        layout="wide"
        )
    st.title("Damage and Loss Model Library")
    render_seismic_tree()
    render_wind_tree()