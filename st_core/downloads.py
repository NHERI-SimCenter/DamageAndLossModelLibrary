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

    get_fragility_csv() -> bytes
        Build and return a fragility CSV filtered to the added components,
        preserving the exact column layout of each source fragility.csv.

    get_consequence_csv() -> bytes
        Build and return a consequence_repair CSV filtered to the added
        components (all consequence types: Cost, Time, Carbon, Energy),
        preserving the exact column layout of each source CSV.

    render_download_buttons()
        Render Streamlit download buttons for all available download files.

Usage
-----
    from st_core.downloads import render_download_buttons

    render_download_buttons()
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path
from typing import Optional

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


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8-encoded CSV bytes."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _group_entries_by_json_path(entries: list) -> dict[str, list[str]]:
    """
    Group added-component IDs by their source ``json_path``.

    Returns a mapping of ``{json_path: [comp_id, ...]}``.  Components that
    share the same library directory are batched together so each CSV file
    is opened only once.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        groups[entry["json_path"]].append(entry["comp_id"])
    return dict(groups)


@st.cache_data(show_spinner=False)
def _load_raw_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV from *csv_path* with ``ID`` as a plain string column.

    Results are cached by Streamlit so each file is read from disk at most
    once per session.  Returns ``None`` if the file does not exist or cannot
    be parsed.
    """
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, dtype=str)
    except Exception:
        return None


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

    return _df_to_csv_bytes(df)


def get_fragility_csv() -> bytes:
    """
    Build a fragility CSV filtered to the current added-components list.

    For each unique source library (identified by ``json_path``), the
    sibling ``fragility.csv`` is loaded and rows matching the selected
    component IDs are extracted.  All matched rows are concatenated and
    returned, preserving the original column layout of the source files.

    Components whose IDs are not found in their source file are silently
    skipped.

    Returns
    -------
    bytes
        UTF-8-encoded CSV content, or an empty CSV if no data is found.
    """
    entries: list = st.session_state.get(_ADDED_KEY, [])
    groups = _group_entries_by_json_path(entries)

    frames: list[pd.DataFrame] = []
    for json_path, comp_ids in groups.items():
        csv_path = str(Path(json_path).parent / "fragility.csv")
        df = _load_raw_csv(csv_path)
        if df is None:
            continue
        matched = df[df["ID"].isin(comp_ids)]
        if not matched.empty:
            frames.append(matched)

    if not frames:
        # Return an empty CSV with just the header from the first available
        # source, or a minimal stub if none could be loaded.
        for json_path in groups:
            csv_path = str(Path(json_path).parent / "fragility.csv")
            df = _load_raw_csv(csv_path)
            if df is not None:
                return _df_to_csv_bytes(df.iloc[0:0])
        return _df_to_csv_bytes(pd.DataFrame(columns=["ID"]))

    result = pd.concat(frames, ignore_index=True)
    return _df_to_csv_bytes(result)


def get_consequence_csv() -> bytes:
    """
    Build a consequence_repair CSV filtered to the current added-components list.

    Each component contributes up to four rows in the consequence file
    (Cost, Time, Carbon, Energy), identified by the ``{comp_id}-{Type}``
    pattern in the ``ID`` column.  Rows are matched using a prefix filter
    so all consequence types are included automatically.

    For each unique source library the sibling ``consequence_repair.csv``
    is loaded and filtered.  All matched rows are concatenated and returned,
    preserving the original column layout.

    Components with no consequence data in their source file are silently
    skipped.

    Returns
    -------
    bytes
        UTF-8-encoded CSV content, or an empty CSV if no data is found.
    """
    entries: list = st.session_state.get(_ADDED_KEY, [])
    groups = _group_entries_by_json_path(entries)

    frames: list[pd.DataFrame] = []
    for json_path, comp_ids in groups.items():
        csv_path = str(Path(json_path).parent / "consequence_repair.csv")
        df = _load_raw_csv(csv_path)
        if df is None:
            continue
        # Match rows whose ID starts with "{comp_id}-" to capture all
        # consequence types (Cost, Time, Carbon, Energy) for each component.
        prefixes = tuple(f"{cid}-" for cid in comp_ids)
        matched = df[df["ID"].str.startswith(prefixes)]
        if not matched.empty:
            frames.append(matched)

    if not frames:
        for json_path in groups:
            csv_path = str(Path(json_path).parent / "consequence_repair.csv")
            df = _load_raw_csv(csv_path)
            if df is not None:
                return _df_to_csv_bytes(df.iloc[0:0])
        return _df_to_csv_bytes(pd.DataFrame(columns=["ID"]))

    result = pd.concat(frames, ignore_index=True)
    return _df_to_csv_bytes(result)


# ─── Streamlit renderers ───────────────────────────────────────────────────────

def render_download_buttons() -> None:
    """
    Render download buttons for all available download files.

    Provides:
    - **component_quantity.csv** — one row per added component; ID column
      populated, all other columns blank.
    - **fragility.csv** — rows from the source fragility data matching the
      added components, preserving the original column layout.
    - **consequence_repair.csv** — rows from the source consequence data
      matching the added components (all consequence types), preserving the
      original column layout.
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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="⬇️ component_quantity.csv",
            data=get_component_quantity_csv(),
            file_name="component_quantity.csv",
            mime="text/csv",
            width='stretch',
            help=(
                "One row per added component. "
                "ID column is populated; remaining columns are blank."
            ),
        )

    with col2:
        st.download_button(
            label="⬇️ fragility.csv",
            data=get_fragility_csv(),
            file_name="fragility.csv",
            mime="text/csv",
            width='stretch',
            help=(
                "Fragility data for the selected components, "
                "in the original source format."
            ),
        )

    with col3:
        st.download_button(
            label="⬇️ consequence_repair.csv",
            data=get_consequence_csv(),
            file_name="consequence_repair.csv",
            mime="text/csv",
            width='stretch',
            help=(
                "Consequence repair data (Cost, Time, Carbon, Energy) "
                "for the selected components, in the original source format."
            ),
        )