import streamlit as st
from st_ui.auth_ui import render_login_panel

def render_sidebar() -> None:
    """
    Render the sidebar
    """
    with st.sidebar:
        render_login_panel()