import streamlit as st
from dlml.web.st_search.search_ui import render_search_and_library
from dlml.web.st_ui.branding import render_top_bar
from dlml.web.st_ui.theme import use_full_width


def render_main_page() -> None:
    """
    Render the Browse & Search page: the sticky brand bar, the component search
    panel, and the browse tree.

    The hero is intentionally omitted here — the brand bar already shows the
    "DLML Explorer" name and logos, so the tool goes straight to work and saves
    the vertical space. The full hero stays on the About landing page.

    ``st.set_page_config`` is intentionally NOT called here — it must be the
    first Streamlit command of the run and is set in ``app.py`` before the
    sidebar renders.
    """
    # Data-dense page — use the full page width on large monitors (the base
    # stylesheet caps content for readability, which we lift here).
    use_full_width()
    render_top_bar()
    render_search_and_library()
