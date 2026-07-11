import streamlit as st
from st_search.search_ui import render_search_and_library
from st_ui.branding import render_header
from st_ui.theme import use_full_width


def render_main_page() -> None:
    """
    Render the main page: the header (title + contributor logos on one line),
    the component search panel, and the browse tree.

    ``st.set_page_config`` is intentionally NOT called here — it must be the
    first Streamlit command of the run and is set in ``app.py`` before the
    sidebar renders.
    """
    # Data-dense page — use the full page width on large monitors (the base
    # stylesheet caps content for readability, which we lift here).
    use_full_width()
    render_header("DLML Explorer", subtitle="Damage and Loss Model Library")
    render_search_and_library()
