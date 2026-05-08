"""
component.py
------------
Standalone display module for Level 5 component leaf nodes.

Provides three public functions for rendering individual fragility-library
component records outside (or inside) the collapsible tree:

    render_component_leaf(comp_id, comp_data, json_path, *, key_prefix="")
        Full seismic leaf: metadata, damage states, fragility curves, and
        consequence curves.

    render_wind_component_leaf(comp_id, comp_data, json_path, *, key_prefix="")
        Wind leaf: metadata, damage states, and fragility curve only
        (SimCenter Wind library carries no consequence data).

    render_component_leaf_button(comp_id, comp_data, json_path, *, key_prefix="", hazard="seismic")
        Drop-in wrapper that replicates the "Load details" session-state guard
        used inside the tree, so the same lazy-loading pattern can be applied
        when embedding a leaf anywhere in the app.

Leaf detail renderers
---------------------
    _render_component_detail(comp_id, comp_data, json_path)
        Internal seismic detail panel — called by render_component_leaf.

    _render_wind_component_detail(comp_id, comp_data, json_path)
        Internal wind detail panel — called by render_wind_component_leaf.

Data helpers (re-exported from tree_visuals for convenience)
------------------------------------------------------------
    _load_fragility_df, _load_consequence_df, _make_fragility_figure,
    _make_consequence_figure, _render_consequence_tab
    — imported directly so callers that already import tree_visuals are not
      forced to change their import paths.

Usage
-----
    from component import render_component_leaf, render_wind_component_leaf
    from component import render_component_leaf_button

    # Render a seismic component directly (no lazy-load gate):
    render_component_leaf("B.10.31.001a", comp_data_dict, "/path/to/fragility.json")

    # Render with the lazy-load button guard (matches tree behaviour):
    render_component_leaf_button(
        "B.10.31.001a", comp_data_dict, "/path/to/fragility.json",
        key_prefix="my_page_", hazard="seismic",
    )
"""

from __future__ import annotations

import json
from typing import Dict, List

import streamlit as st

from st_visuals.figures import make_consequence_figure, make_fragility_figure
from st_visuals.helpers_visual import load_consequence_df, load_fragility_df

# Consequence type options shown in the selectbox
_C_TYPES: List[str] = ["Cost", "Time", "Carbon", "Energy"]

# ─── Internal: consequence tab ─────────────────────────────────────────────────

def _render_consequence_tab(comp_id: str, json_path: str) -> None:
    """
    Render the consequence-curves tab for a single seismic component.

    Checks which consequence types are available in ``consequence_repair.csv``
    and presents a radio selector before plotting.  Falls back to an
    informational message when no data exists.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"B.10.31.001a"``.
    json_path : str
        Path to the source ``fragility.json``; used to locate
        ``consequence_repair.csv`` in the same directory.
    """
    repair_df = load_consequence_df(json_path)

    if repair_df is None:
        st.info(
            "No consequence data file found for this source directory.",
            icon="ℹ️",
        )
        return

    lvl0 = repair_df.index.get_level_values(0)
    if comp_id not in lvl0:
        st.info(
            f"No consequence records found for `{comp_id}` in the repair database.",
            icon="ℹ️",
        )
        return

    available_types = [t for t in _C_TYPES if t in repair_df.loc[comp_id].index]

    if not available_types:
        st.info("No consequence types available for this component.", icon="ℹ️")
        return

    c_type = st.radio(
        "Consequence type",
        options=available_types,
        horizontal=True,
        key=f"cons_type_{comp_id}",
    )

    st.plotly_chart(
        make_consequence_figure(comp_id, c_type, json_path),
        use_container_width=True,
        key=f"cons_{comp_id}_{c_type}",
    )


# ─── Internal: seismic detail panel ───────────────────────────────────────────

def _render_component_detail(
    comp_id: str,
    comp_data: dict,
    json_path: str,
) -> None:
    """
    Render the full inline detail panel for a seismic component leaf.

    Displays a two-column layout:
    * Left  — component metadata (ID, block size, integer-qty flag) and an
              expandable list of damage states with repair actions.
    * Right — optional technical-notes expander, then tabbed charts:
              *Fragility curves* and *Consequence curves*.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"B.10.31.001a"``.
    comp_data : dict
        Full component record loaded from ``fragility.json``.
    json_path : str
        Path to the source ``fragility.json``.  Passed to consequence helpers
        so they can locate ``consequence_repair.csv`` in the same directory.
    """
    description = comp_data.get("Description", "")
    comments = comp_data.get("Comments", "")
    block_size = comp_data.get("SuggestedComponentBlockSize", "")
    round_up = comp_data.get("RoundUpToIntegerQuantity", "")
    limit_states: dict = comp_data.get("LimitStates", {})

    col_left, col_right = st.columns([1, 2], gap="large")

    # ── Left: metadata + damage states ────────────────────────────────────
    with col_left:
        st.markdown("**Component metadata**")

        for label, value in [
            ("ID", f"`{comp_id}`"),
            ("Block size", f"`{block_size}`" if block_size else "—"),
            ("Integer qty", round_up if round_up else "—"),
        ]:
            c1, c2 = st.columns([1, 1])
            c1.caption(label)
            c2.caption(value)

        if description:
            st.caption(f"_{description}_")

        st.divider()
        st.markdown("**Damage states**")

        total_ds = sum(len(ds_dict) for ds_dict in limit_states.values())
        if total_ds:
            for ls_key, ls_data in limit_states.items():
                for ds_key, ds_data in ls_data.items():
                    desc_text = (
                        ds_data.get("Description", "No description.")
                        if isinstance(ds_data, dict)
                        else str(ds_data)
                    )
                    with st.expander(f"{ls_key} / {ds_key}", expanded=False):
                        st.caption(desc_text)
                        if isinstance(ds_data, dict) and ds_data.get("RepairAction"):
                            st.caption(
                                f"**Repair action:** {ds_data['RepairAction']}"
                            )
        else:
            st.caption("No limit-state data found.")

    # ── Right: comments + charts ───────────────────────────────────────────
    with col_right:
        if comments:
            with st.expander("Technical notes / comments", expanded=False):
                st.caption(comments)

        tab_frag, tab_cons = st.tabs(["Fragility curves", "Consequence curves"])

        with tab_frag:
            frag_df = load_fragility_df(json_path)
            if frag_df is not None and comp_id in frag_df.index:
                csv_row = frag_df.loc[comp_id]
                csv_row_flat = {
                    f"{a}-{b}" if b else str(a): v
                    for (a, b), v in csv_row.items()
                }
                st.plotly_chart(
                    make_fragility_figure(
                        comp_id,
                        json.dumps(limit_states),
                        json.dumps(csv_row_flat, default=str),
                    ),
                    use_container_width=True,
                    key=f"frag_{comp_id}",
                )
            else:
                st.info(
                    "No fragility data available to generate curves.", icon="ℹ️"
                )

        with tab_cons:
            _render_consequence_tab(comp_id, json_path)


# ─── Internal: wind detail panel ──────────────────────────────────────────────

def _render_wind_component_detail(
    comp_id: str,
    comp_data: dict,
    json_path: str,
) -> None:
    """
    Render the inline detail panel for a wind library component leaf.

    Mirrors ``_render_component_detail`` but omits the consequence tab because
    the SimCenter Wind Component Library carries no consequence data.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"DOOR.garage.001a"``.
    comp_data : dict
        Full component record from ``fragility.json``.
    json_path : str
        Path to the source ``fragility.json``.
    """
    description: str = comp_data.get("Description", "")
    comments: str = comp_data.get("Comments", "")
    references: list = comp_data.get("Reference", [])
    block_size: str = comp_data.get("SuggestedComponentBlockSize", "")
    limit_states: dict = comp_data.get("LimitStates", {})

    col_left, col_right = st.columns([1, 2])

    # ── Left: metadata ─────────────────────────────────────────────────────
    with col_left:
        if description:
            st.markdown(f"**Description:** {description}")

        if block_size:
            st.caption(f"Block size: `{block_size}`")

        if references:
            st.caption(
                "References: " + ", ".join(f"`{r}`" for r in references)
            )

        if limit_states:
            st.markdown("**Limit states / damage states**")
            for ls_key, ls_data in limit_states.items():
                if not isinstance(ls_data, dict):
                    continue
                for ds_key, ds_data in ls_data.items():
                    desc_text = (
                        ds_data.get("Description", "")
                        if isinstance(ds_data, dict)
                        else str(ds_data)
                    )
                    with st.expander(f"{ls_key} / {ds_key}", expanded=False):
                        st.caption(desc_text)
        else:
            st.caption("No limit-state data found.")

    # ── Right: comments + fragility chart ─────────────────────────────────
    with col_right:
        if comments:
            with st.expander("Technical notes / comments", expanded=False):
                st.caption(comments)

        frag_df = load_fragility_df(json_path)
        if frag_df is not None and comp_id in frag_df.index:
            csv_row = frag_df.loc[comp_id]
            csv_row_flat = {
                f"{a}-{b}" if b else str(a): v
                for (a, b), v in csv_row.items()
            }
            st.plotly_chart(
                make_fragility_figure(
                    comp_id,
                    json.dumps(limit_states),
                    json.dumps(csv_row_flat, default=str),
                ),
                use_container_width=True,
                key=f"wind_frag_{comp_id}",
            )
        else:
            st.info(
                "No fragility data available to generate curves.", icon="ℹ️"
            )


# ─── Public API ────────────────────────────────────────────────────────────────

def render_component_leaf(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    *,
    key_prefix: str = "",
) -> None:
    """
    Render the full detail panel for a seismic component leaf node.

    This is the standalone version of the Level 5 leaf content — no
    ``st.expander`` wrapper, no session-state load gate.  Use this when you
    want to embed a component panel directly on a page or inside your own
    container.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"B.10.31.001a"``.
    comp_data : dict
        Full component record from ``fragility.json``.
    json_path : str
        Absolute path to the source ``fragility.json``.
    key_prefix : str, optional
        Prepended to all Streamlit widget keys to prevent key collisions when
        multiple leaves are rendered on the same page.
    """
    _render_component_detail(comp_id, comp_data, json_path)


def render_wind_component_leaf(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    *,
    key_prefix: str = "",
) -> None:
    """
    Render the detail panel for a wind / hurricane component leaf node.

    Displays description, block size, references, limit states, and a
    fragility curve.  Consequence curves are omitted because the SimCenter
    Wind Component Library does not include consequence data.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"DOOR.garage.001a"``.
    comp_data : dict
        Full component record from ``fragility.json``.
    json_path : str
        Absolute path to the source ``fragility.json``.
    key_prefix : str, optional
        Prepended to all Streamlit widget keys to prevent key collisions.
    """
    _render_wind_component_detail(comp_id, comp_data, json_path)


def render_component_leaf_button(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    *,
    key_prefix: str = "",
    hazard: str = "seismic",
) -> None:
    """
    Render a component leaf with the lazy-load button guard from the tree.

    Replicates the session-state pattern used inside the collapsible tree:
    a "Load details" button is shown on first render; once clicked the detail
    panel appears and is retained across re-runs.  This keeps the Streamlit
    widget tree small until the user explicitly requests the content.

    Parameters
    ----------
    comp_id : str
        Component identifier.
    comp_data : dict
        Full component record from ``fragility.json``.
    json_path : str
        Absolute path to the source ``fragility.json``.
    key_prefix : str, optional
        Prepended to session-state and widget keys for collision-free use on
        multi-component pages.
    hazard : {"seismic", "wind"}, optional
        Which detail renderer to call once loaded.  Defaults to ``"seismic"``.
    """
    load_key = f"{key_prefix}loaded_{comp_id}"
    btn_key = f"{key_prefix}btn_{comp_id}"

    if load_key not in st.session_state:
        if st.button("Load details", key=btn_key, type="secondary"):
            st.session_state[load_key] = True
            st.rerun()
    elif comp_data:
        if hazard == "wind":
            _render_wind_component_detail(comp_id, comp_data, json_path)
        else:
            _render_component_detail(comp_id, comp_data, json_path)
    else:
        st.warning(
            f"Full data for `{comp_id}` was not found in the source file.",
            icon="⚠️",
        )