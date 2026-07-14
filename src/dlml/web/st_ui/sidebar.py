import streamlit as st
from dlml.web.st_core.component import render_added_components_list


def render_sidebar() -> None:
    """
    Render the sidebar.

    Reserved entirely for the user's selected models — navigation and the
    light/dark switch live in the top bar (see
    :func:`dlml.web.st_ui.branding.render_top_bar`). A short intro explains the
    panel, then :func:`render_added_components_list` renders the selection and
    the download tools.
    """
    with st.sidebar:
        st.markdown("### 🗂️ Selected models")
        st.caption(
            "Models you add from **Browse & Search** are collected here. Keep "
            "them for side-by-side review as you explore, then download their "
            "data for your project. Your selection follows you across pages. "
            "You can hide this panel any time with the **«** at its top-right "
            "corner."
        )
        st.divider()
        render_added_components_list()
