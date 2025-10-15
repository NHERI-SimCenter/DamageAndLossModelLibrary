import streamlit as st
from auth.simple_auth import ensure_login, current_user, logout_button

def render_login_panel() -> None:
    """
    login UI
    """
    ensure_login()

