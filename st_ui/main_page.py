import streamlit as st
from st_search.search_ui import render_search_and_library


def render_main_page() -> None:
    """
    Render the main page: the component search panel and the browse tree.

    ``st.set_page_config`` is intentionally NOT called here — it must be the
    first Streamlit command of the run and is set in ``app.py`` before the
    sidebar renders.
    """
    st.title("Damage and Loss Model Library")
    render_search_and_library()
