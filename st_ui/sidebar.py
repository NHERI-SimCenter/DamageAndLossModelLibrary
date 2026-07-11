import streamlit as st
from st_ui.theme import render_theme_toggle
from st_core.component import render_added_components_list

def render_sidebar() -> None:
    """
    Render the sidebar
    """
    with st.sidebar:
        render_theme_toggle()
        st.divider()
        render_added_components_list()