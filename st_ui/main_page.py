import streamlit as st
from auth.login_ui import render_login_panel
from st_search.fuzzy_visuals import render_fuzzy_search
from figures import render_selected_fragility_figures
from st_search.tree_visuals import render_seismic_tree


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
    # render_fuzzy_search()
    render_seismic_tree()
    render_selected_fragility_figures()
