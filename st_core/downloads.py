"""
downloads.py
------------
Manages functions that let the user download data from their selected
components.

Public functions
----------------
    get_component_quantity_csv() -> bytes
        Build and return a component_quantity CSV for the user's current
        added-components list.  The ID column is populated from session
        state; all other columns are left blank (to be implemented).

    render_download_buttons()
        Render Streamlit download buttons for all available download files.

Usage
-----
    from st_core.downloads import render_download_buttons

    render_download_buttons()
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

# Session-state key used by component.py for the added-components list.
_ADDED_KEY = "added_components"

# Columns matching the component_quantity.csv format.
_COMPONENT_QUANTITY_COLUMNS = [
    "ID",
    "Units",
    "Location",
    "Direction",
    "Theta_0",
    "Blocks",
    "Family",
    "Theta_1",
    "Comment",
]


# ─── Data builders ─────────────────────────────────────────────────────────────

def get_component_quantity_csv() -> bytes:
    """
    Build a component_quantity CSV for the current added-components list.

    Only the ``ID`` column is populated; all other columns are left blank.
    The returned bytes are suitable for use directly with
    ``st.download_button``.

    Returns
    -------
    bytes
        UTF-8-encoded CSV content.
    """
    entries: list = st.session_state.get(_ADDED_KEY, [])
    ids = [entry["comp_id"] for entry in entries]

    df = pd.DataFrame(columns=_COMPONENT_QUANTITY_COLUMNS)
    df["ID"] = ids

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# ─── Streamlit renderers ───────────────────────────────────────────────────────

def render_download_buttons() -> None:
    """
    Render download buttons for all available download files.

    Currently provides:
    - **component_quantity.csv** — one row per added component with the ID
      column filled in and all other columns blank.

    Additional download types will be added here as they are implemented.
    """
    entries: list = st.session_state.get(_ADDED_KEY, [])

    if not entries:
        st.info(
            "No components added yet. Add components using the "
            "**➕ Add component** button to enable downloads.",
            icon="📥",
        )
        return

    st.markdown(f"### 📥 Downloads  ·  `{len(entries)}` component(s)")

    st.download_button(
        label="⬇️ component_quantity.csv",
        data=get_component_quantity_csv(),
        file_name="component_quantity.csv",
        mime="text/csv",
        help=(
            "CSV with one row per added component. "
            "ID column is populated; remaining columns are blank."
        ),
    )