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

    render_added_components_list()
        Render the list of all components the user has added via the
        "Add component" button.  Each entry is shown in a collapsible
        expander using the appropriate detail renderer (seismic or wind),
        with a "Remove" button to delete it from the list.

Leaf detail renderers
---------------------
    _render_component_detail(comp_id, comp_data, json_path, *, key_prefix="")
        Internal seismic detail panel — called by render_component_leaf.

    _render_wind_component_detail(comp_id, comp_data, json_path, *, key_prefix="")
        Internal wind detail panel — called by render_wind_component_leaf.

Usage
-----
    from st_core.component import render_component_leaf, render_wind_component_leaf
    from st_core.component import render_component_leaf_button
    from st_core.component import render_added_components_list

    # Render a seismic component directly (no lazy-load gate):
    render_component_leaf("B.10.31.001a", comp_data_dict, "/path/to/fragility.json")

    # Render with the lazy-load button guard (matches tree behaviour):
    render_component_leaf_button(
        "B.10.31.001a", comp_data_dict, "/path/to/fragility.json",
        key_prefix="my_page_", hazard="seismic",
    )

    # Render the running list of added components (place anywhere on the page):
    render_added_components_list()
"""

from __future__ import annotations

import json
from typing import List, Optional

import streamlit as st

from st_visuals.figures import make_consequence_figure, make_fragility_figure
from st_visuals.helpers_visual import load_consequence_df, load_fragility_df
from st_core.downloads import render_download_buttons

# Consequence type options shown in the selectbox
_C_TYPES: List[str] = ["Cost", "Time", "Carbon", "Energy"]

# Session-state key for the added-components list
_ADDED_KEY = "added_components"


# ─── Session-state helpers ─────────────────────────────────────────────────────

def _initialize_added_components_state() -> None:
    """
    Ensure the added-components list exists in ``st.session_state``.

    Each entry is a dict with keys:
        ``comp_id``   – component identifier string
        ``comp_data`` – full component record dict
        ``json_path`` – path to the source fragility.json
        ``hazard``    – ``"seismic"`` or ``"wind"``
    """
    if _ADDED_KEY not in st.session_state:
        st.session_state[_ADDED_KEY] = []


def _is_component_added(comp_id: str, json_path: Optional[str] = None) -> bool:
    """
    Return True if the record is already in the added-components list.

    The dedup key is ``(comp_id, json_path)`` so a consequence record and a
    fragility component that share an ID (e.g. FEMA ``B.10.31.001`` in both
    fragility.json and consequence_repair.json) are tracked independently. When
    *json_path* is None, any entry matching *comp_id* counts.
    """
    _initialize_added_components_state()
    return any(
        entry["comp_id"] == comp_id
        and (json_path is None or entry.get("json_path") == json_path)
        for entry in st.session_state[_ADDED_KEY]
    )


# ─── Public selection API ──────────────────────────────────────────────────────
# Shared so any surface (tree, search results, …) adds to the SAME session-state
# list the sidebar and downloads read from.

def is_component_added(comp_id: str, json_path: Optional[str] = None) -> bool:
    """Public: True if the record is already in the added-components list."""
    return _is_component_added(comp_id, json_path)


def add_component(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    hazard: str,
    kind: str = "fragility",
) -> bool:
    """
    Add a record to the shared added-components list.

    Parameters
    ----------
    comp_id, comp_data, json_path : see :func:`render_component_leaf`.
    hazard : {"seismic", "wind"}
        Used to pick the detail renderer for fragility records. Note this is
        ``"wind"`` (not ``"hurricane"``) to match the existing list contract.
    kind : {"fragility", "consequence"}
        Which detail renderer the sidebar uses, and how downloads treat it.
        Consequence records (e.g. Hazus occupancy classes) have no fragility row
        but do contribute to consequence_repair.csv.

    Returns
    -------
    bool
        True if the record was added, False if it was already present.
    """
    _initialize_added_components_state()
    if _is_component_added(comp_id, json_path):
        return False
    st.session_state[_ADDED_KEY].append(
        {
            "comp_id": comp_id,
            "comp_data": comp_data,
            "json_path": json_path,
            "hazard": hazard,
            "kind": kind,
        }
    )
    return True


# ─── Internal: consequence tab ─────────────────────────────────────────────────

def _render_consequence_tab(
    comp_id: str,
    json_path: str,
    *,
    key_prefix: str = "",
) -> None:
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
    key_prefix : str, optional
        Prepended to all widget keys to prevent collisions when the same
        component is rendered more than once on the page.
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
        key=f"{key_prefix}cons_type_{comp_id}",
    )

    st.plotly_chart(
        make_consequence_figure(comp_id, c_type, json_path),
        use_container_width=True,
        key=f"{key_prefix}cons_{comp_id}_{c_type}",
    )


# ─── Internal: add-component button ───────────────────────────────────────────

def _render_add_button(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    hazard: str,
    *,
    key_prefix: str = "",
    kind: str = "fragility",
) -> None:
    """
    Render the "Add component" button at the bottom of a detail panel.

    If the record is already in the added list the button is replaced by a
    muted caption so the user has clear feedback without a disabled widget.

    Parameters
    ----------
    comp_id : str
        Component identifier.
    comp_data : dict
        Full record from ``fragility.json`` or ``consequence_repair.json``.
    json_path : str
        Path to the source JSON (also the dedup key alongside *comp_id*).
    hazard : {"seismic", "wind"}
        Hazard type (picks the fragility detail renderer in the list).
    key_prefix : str, optional
        Prepended to the button widget key to prevent collisions.
    kind : {"fragility", "consequence"}
        Record kind, forwarded to :func:`add_component`.
    """
    _initialize_added_components_state()
    st.divider()

    if _is_component_added(comp_id, json_path):
        st.caption(f"✅ `{comp_id}` is already in your component list.")
    else:
        if st.button(
            "➕ Add component",
            key=f"{key_prefix}add_btn_{comp_id}",
            type="primary",
            help="Add this record to the page list for side-by-side review and download.",
        ):
            add_component(comp_id, comp_data, json_path, hazard, kind=kind)
            st.rerun()


# ─── Internal: fragility chart + consequence availability ─────────────────────

def _has_consequence_data(comp_id: str, json_path: str) -> bool:
    """
    True if the source's consequence_repair.csv has a record for *comp_id*.

    Hazus sources key repair consequences by occupancy class (RES1, COM1, …)
    rather than by fragility component, so their components have no
    per-component consequence curve — in that case the Consequence tab is
    omitted instead of showing an empty "no records" message.
    """
    repair_df = load_consequence_df(json_path)
    if repair_df is None:
        return False
    return comp_id in repair_df.index.get_level_values(0)


def _render_fragility_chart(
    comp_id: str,
    json_path: str,
    limit_states: dict,
    *,
    key_prefix: str = "",
) -> None:
    """Render the fragility-curve plot for a single component."""
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
            key=f"{key_prefix}frag_{comp_id}",
        )
    else:
        st.info("No fragility data available to generate curves.", icon="ℹ️")


# ─── Internal: seismic detail panel ───────────────────────────────────────────

def _render_component_detail(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    *,
    key_prefix: str = "",
) -> None:
    """
    Render the full inline detail panel for a seismic component leaf.

    Displays a two-column layout:
    * Left  — component metadata (ID, block size, integer-qty flag) and an
              expandable list of damage states with repair actions.
    * Right — "Add component" button, optional technical-notes expander,
              then tabbed charts: *Fragility curves* and *Consequence curves*.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"B.10.31.001a"``.
    comp_data : dict
        Full component record loaded from ``fragility.json``.
    json_path : str
        Path to the source ``fragility.json``.
    key_prefix : str, optional
        Prepended to all widget keys and expander labels to prevent collisions
        when the same component is rendered more than once on the page (e.g.
        simultaneously in the tree and in the added-components list).
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
                    # key_prefix in the label prevents duplicate-label warnings
                    # when this component is rendered in two places at once.
                    with st.expander(
                        f"{key_prefix}{ls_key} / {ds_key}", expanded=False
                    ):
                        st.caption(desc_text)
                        if isinstance(ds_data, dict) and ds_data.get("RepairAction"):
                            st.caption(
                                f"**Repair action:** {ds_data['RepairAction']}"
                            )
        else:
            st.caption("No limit-state data found.")

    # ── Right: add button + comments + charts ─────────────────────────────
    with col_right:
        _render_add_button(
            comp_id, comp_data, json_path, hazard="seismic", key_prefix=key_prefix
        )

        if comments:
            with st.expander(
                f"{key_prefix}Technical notes / comments", expanded=False
            ):
                st.caption(comments)

        # Only show the Consequence tab when this component actually has
        # consequence records. Hazus sources key consequences by occupancy
        # class, not by component, so the tab is omitted there (rather than
        # showing a confusing "no records found" message).
        if _has_consequence_data(comp_id, json_path):
            tab_frag, tab_cons = st.tabs(["Fragility curves", "Consequence curves"])
            with tab_frag:
                _render_fragility_chart(
                    comp_id, json_path, limit_states, key_prefix=key_prefix
                )
            with tab_cons:
                _render_consequence_tab(comp_id, json_path, key_prefix=key_prefix)
        else:
            st.markdown("**Fragility curves**")
            _render_fragility_chart(
                comp_id, json_path, limit_states, key_prefix=key_prefix
            )
            st.caption(
                "No component-level consequence (repair) curves are available "
                "for this component in this source."
            )


# ─── Internal: wind detail panel ──────────────────────────────────────────────

def _render_wind_component_detail(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    *,
    key_prefix: str = "",
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
    key_prefix : str, optional
        Prepended to all widget keys and expander labels to prevent collisions
        when the same component is rendered more than once on the page.
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
                    with st.expander(
                        f"{key_prefix}{ls_key} / {ds_key}", expanded=False
                    ):
                        st.caption(desc_text)
        else:
            st.caption("No limit-state data found.")

    # ── Right: add button + comments + fragility chart ─────────────────────
    with col_right:
        _render_add_button(
            comp_id, comp_data, json_path, hazard="wind", key_prefix=key_prefix
        )

        if comments:
            with st.expander(
                f"{key_prefix}Technical notes / comments", expanded=False
            ):
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
                key=f"{key_prefix}wind_frag_{comp_id}",
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
    _render_component_detail(comp_id, comp_data, json_path, key_prefix=key_prefix)


def render_wind_component_leaf(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    *,
    key_prefix: str = "",
) -> None:
    """
    Render the detail panel for a wind / hurricane component leaf node.

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
    _render_wind_component_detail(comp_id, comp_data, json_path, key_prefix=key_prefix)


def render_consequence_leaf(
    comp_id: str,
    comp_data: dict,
    json_path: str,
    *,
    key_prefix: str = "",
) -> None:
    """
    Render the detail panel for a consequence_repair leaf node.

    Consequence records are keyed by component (FEMA P-58) or by occupancy class
    (Hazus, e.g. ``STR.RES1``). Shows metadata and damage states on the left and
    the repair-consequence curves (Cost / Time / …) on the right, with an "Add"
    button so the record can be included in the consequence_repair.csv download.

    Parameters
    ----------
    comp_id : str
        Consequence identifier, e.g. ``"STR.RES1"`` or ``"B.10.31.001"``.
    comp_data : dict
        Full record from ``consequence_repair.json``.
    json_path : str
        Path to the source ``consequence_repair.json`` (its sibling
        ``consequence_repair.csv`` supplies the curves).
    key_prefix : str, optional
        Prepended to widget keys to prevent collisions.
    """
    description = comp_data.get("Description", "")
    comments = comp_data.get("Comments", "")
    block_size = comp_data.get("SuggestedComponentBlockSize", "")
    damage_states: dict = comp_data.get("DamageStates", {})

    col_left, col_right = st.columns([1, 2], gap="large")

    # ── Left: metadata + damage states ────────────────────────────────────
    with col_left:
        st.markdown("**Consequence metadata**")

        for label, value in [
            ("ID", f"`{comp_id}`"),
            ("Block size", f"`{block_size}`" if block_size else "—"),
        ]:
            c1, c2 = st.columns([1, 1])
            c1.caption(label)
            c2.caption(value)

        if description:
            st.caption(f"_{description}_")

        st.divider()
        st.markdown("**Damage states**")

        if damage_states:
            for ds_key, ds_data in damage_states.items():
                desc_text = (
                    ds_data.get("Description", "No description.")
                    if isinstance(ds_data, dict)
                    else str(ds_data)
                )
                with st.expander(f"{key_prefix}{ds_key}", expanded=False):
                    st.caption(desc_text)
        else:
            st.caption("No damage-state data found.")

    # ── Right: add button + comments + consequence curves ─────────────────
    with col_right:
        _render_add_button(
            comp_id, comp_data, json_path, hazard="seismic",
            key_prefix=key_prefix, kind="consequence",
        )

        if comments:
            with st.expander(
                f"{key_prefix}Technical notes / comments", expanded=False
            ):
                st.caption(comments)

        st.markdown("**Repair consequence curves**")
        _render_consequence_tab(comp_id, json_path, key_prefix=key_prefix)


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
            _render_wind_component_detail(
                comp_id, comp_data, json_path, key_prefix=key_prefix
            )
        else:
            _render_component_detail(
                comp_id, comp_data, json_path, key_prefix=key_prefix
            )
    else:
        st.warning(
            f"Full data for `{comp_id}` was not found in the source file.",
            icon="⚠️",
        )


def render_added_components_list() -> None:
    """
    Render the list of components added via the "Add component" button.

    Each component is shown in a collapsible expander whose header contains
    the component ID and a short description preview.  The appropriate detail
    renderer is called with a unique ``key_prefix`` derived from the slot
    index so that widget keys never collide — even if the same component
    appears in both the tree and this list simultaneously.

    A **🗑️ Remove** button appears at the top of each expanded panel, and a
    **🗑️ Clear all** button sits below the list.  If the list is empty, an
    info message is shown instead.

    Usage
    -----
        from st_core.component import render_added_components_list
        render_added_components_list()
    """
    _initialize_added_components_state()
    entries: list = st.session_state[_ADDED_KEY]

    if not entries:
        st.info(
            "No components added yet. Use the **➕ Add component** button "
            "inside any component detail panel.",
            icon="📋",
        )
        return

    st.markdown(f"### 📋 Added Components  `{len(entries)}`")

    for idx, entry in enumerate(entries):
        comp_id: str = entry["comp_id"]
        comp_data: dict = entry["comp_data"]
        json_path: str = entry["json_path"]
        hazard: str = entry["hazard"]
        kind: str = entry.get("kind", "fragility")

        # Unique prefix per list slot — prevents any key collision with the
        # tree or with other slots that happen to share the same comp_id.
        slot_prefix = f"list{idx}_"

        raw_desc: str = comp_data.get("Description", "")
        preview = raw_desc[:80] + "…" if len(raw_desc) > 80 else raw_desc
        badge = "🧾" if kind == "consequence" else ("🌊" if hazard == "wind" else "🌍")

        with st.expander(
            f"{badge}  **{comp_id}**  ·  {preview}",
            expanded=False,
        ):
            # Remove button at the top of the expanded panel
            _, btn_col = st.columns([5, 1])
            with btn_col:
                if st.button(
                    "🗑️ Remove",
                    key=f"list_remove_{comp_id}_{idx}",
                    type="secondary",
                    help=f"Remove {comp_id} from the list",
                ):
                    st.session_state[_ADDED_KEY].pop(idx)
                    st.rerun()

            # Full detail panel rendered with a unique key_prefix.
            # The add button inside will show "✅ already added" since the
            # record is already present in the list.
            if kind == "consequence":
                render_consequence_leaf(
                    comp_id, comp_data, json_path, key_prefix=slot_prefix
                )
            elif hazard == "wind":
                _render_wind_component_detail(
                    comp_id, comp_data, json_path, key_prefix=slot_prefix
                )
            else:
                _render_component_detail(
                    comp_id, comp_data, json_path, key_prefix=slot_prefix
                )

    st.divider()
    if st.button(
        "🗑️ Clear all",
        key="added_components_clear_all",
        type="secondary",
    ):
        st.session_state[_ADDED_KEY].clear()
        st.rerun()
    
    render_download_buttons()