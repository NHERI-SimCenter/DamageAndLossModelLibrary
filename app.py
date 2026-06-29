import streamlit as st

# set_page_config must be the first Streamlit command of the run, before the
# sidebar (which renders widgets) executes.
st.set_page_config(
    page_title="DLML Explorer",
    page_icon="📊",
    layout="wide",
)

from st_ui.sidebar import render_sidebar
from st_ui.main_page import render_main_page

render_sidebar()
render_main_page()
