"""
tree_visuals.py
---------------
Hierarchical tree view for the fragility component library.

Renders a four-level collapsible tree inside Streamlit:

  Seismic
  └── Source  (FEMA P-58 / Hazus …)
      └── Component Group  (B - Shell / GF - Geotechnical Failure …)
          └── Sub-Group  (B.10.31 - Steel Columns / GF.H - Horizontal Spreading …)
              └── Component  [detail panel + fragility / consequence charts]

Usage
-----
    from tree_visuals import render_seismic_tree
    render_seismic_tree()                          # auto-loads all seismic data
    render_seismic_tree(seismic_objects=my_list)   # pass pre-filtered objects

Performance notes
-----------------
* st.expander executes ALL child code on every Streamlit re-run, whether the
  expander is open or closed. To avoid building hundreds of Plotly figures and
  nested widget trees on every interaction, component detail panels are guarded
  by session-state flags — content is only rendered for explicitly opened leaves.
* _build_tree and _load_full_json are cached so JSON parsing and prefix-routing
  only run once per process lifetime.
* Plotly figures are cached per component ID so they are not rebuilt on re-runs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import colorlover as cl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pelicun.base import convert_to_MultiIndex
from plotly.subplots import make_subplots
from scipy.stats import norm, weibull_min

from st_search.component_search import FuzzyIndex, SearchObject
from st_core.component import _render_wind_component_detail, render_component_leaf, render_wind_component_leaf

from st_visuals.helpers_visual import load_consequence_df



# ─── Palette & constants ───────────────────────────────────────────────────────

_CATEGORY_BADGE: Dict[str, str] = {"FEMA": "FEMA P-58", "HAZUS": "Hazus"}

# Session-state key that holds the set of expanded component IDs
_EXPANDED_KEY = "tree_expanded_components"






# ─── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_cached_index() -> FuzzyIndex:
    """Cache the FuzzyIndex so JSON parsing only runs once per process."""
    return FuzzyIndex()


@st.cache_data(show_spinner=False)
def _load_full_json(json_path: str) -> dict:
    """
    Load and cache the complete fragility.json for a given path.

    The SearchObject only stores component descriptions; this function
    retrieves the full component record (Comments, LimitStates, etc.)
    so the detail panel and fragility charts can be populated.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)





@st.cache_data(show_spinner=False)
def _load_consequence_meta(json_path: str) -> Optional[dict]:
    """
    Load and cache consequence_repair.json metadata from the same directory.

    Returns None if the file is missing or unreadable.
    """
    cons_json = Path(json_path).parent / "consequence_repair.json"
    if not cons_json.exists():
        return None
    try:
        with open(cons_json, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _build_tree(file_paths: tuple[str, ...]) -> Dict[str, dict]:
    """
    Build the nested source → group → sub-group → component tree.

    Accepts a tuple of file paths (hashable for cache_data) so the result
    survives across re-runs without re-routing all component IDs.

    Returns
    -------
    dict
        {
          short_name: {
            "file_path": str,
            "meta": dict,           # _GeneralInformation
            "groups": {
              group_name: {
                "subgroups": {
                  subgroup_name: {
                    "components": [comp_id, ...]
                  }
                }
              }
            }
          }
        }

    Routing logic
    ~~~~~~~~~~~~~
    Each component ID is matched to a sub-group by longest common prefix.
    Example: "GF.H.S" → prefix "GF.H" → "GF.H - Horizontal Spreading".
    Components that match no defined prefix fall into an (Unclassified) bucket.
    """
    tree: Dict[str, dict] = {}

    for fp in file_paths:
        try:
            data = _load_full_json(fp)
        except Exception:
            continue

        meta: dict = data.get("_GeneralInformation", {})
        short_name: str = meta.get("ShortName", Path(fp).parent.name)

        # Build prefix → descriptive-label maps from ComponentGroups.
        #
        # ComponentGroups appears in one of two shapes across the fragility
        # corpus:
        #
        #   (a) dict[str, list[str]] — nested groups with sub-groups. Used by
        #       most Hazus seismic JSONs (e.g. power, water, building).
        #         {
        #           "EP - Electrical Power": [
        #             "EP.S - Substation",
        #             "EP.C - Circuit",
        #             ...
        #           ]
        #         }
        #
        #   (b) list[str] — a flat list of top-level group entries only, with
        #       no sub-group information. Used by the Hazus seismic
        #       Transportation JSON, for example:
        #         ["HRD - Road segments", "HWB - Bridges", "HTU - Tunnels"]
        #
        # Entries in either shape may be bare prefixes (e.g. "DOOR", "STR" —
        # used by the SimCenter Wind library) or descriptive
        # "PREFIX - Description" strings.
        #
        # The render code matches component-ID segments (e.g. "EP", "EP.S")
        # against these maps, so the maps must be keyed by the *bare prefix*.
        # The map values carry the full descriptive string for display in the
        # Level-3 and Level-4 expander headers, matching the pattern the wind
        # tree achieves via _WIND_GROUP_LABELS.
        raw_cg = meta.get("ComponentGroups", {})

        def _split_prefix_label(s: str) -> tuple[str, str]:
            """
            Split a ComponentGroups entry into (prefix, full_label).

            Recognizes both " - " (ASCII hyphen) and " — " (em dash) as the
            separator between the routing prefix and its description. For
            bare-prefix entries with no separator, prefix == label, which
            preserves existing behavior for the wind tree.
            """
            for sep in (" — ", " - "):
                if sep in s:
                    prefix = s.split(sep, 1)[0].strip()
                    return prefix, s
            return s.strip(), s

        group_map: Dict[str, str] = {}
        subgroup_map: Dict[str, str] = {}

        if isinstance(raw_cg, dict):
            # Shape (a): keys are top-level groups, values are sub-group lists.
            for grp_entry, sg_list in raw_cg.items():
                if isinstance(grp_entry, str):
                    prefix, label = _split_prefix_label(grp_entry)
                    group_map[prefix] = label
                if not isinstance(sg_list, list):
                    continue
                for sg_entry in sg_list:
                    if not isinstance(sg_entry, str):
                        continue
                    prefix, label = _split_prefix_label(sg_entry)
                    subgroup_map[prefix] = label
        elif isinstance(raw_cg, list):
            # Shape (b): flat list of top-level group entries only.
            for grp_entry in raw_cg:
                if not isinstance(grp_entry, str):
                    continue
                prefix, label = _split_prefix_label(grp_entry)
                group_map[prefix] = label
            # No sub-group metadata available; sub-group labels will fall back
            # to the bare two-segment prefix in the routing loop below.

        # Collect all real component IDs (skip keys starting with "_")
        comp_ids = [k for k in data if not k.startswith("_")]

        # Route each component into group -> subgroup buckets.
        # Top-level group: first dot-segment of comp_id (e.g. "GF", "STR").
        # Sub-group: first two dot-segments joined (e.g. "GF.H", "STR.W1").
        # When a prefix has no match in the maps (happens for FEMA P-58 IDs
        # whose sub-groups may not be listed), fall back to the bare segment.
        groups: Dict[str, dict] = {}
        for comp_id in comp_ids:
            parts = comp_id.split(".")
            top_segment = parts[0]
            sub_segment = ".".join(parts[:2]) if len(parts) >= 2 else top_segment

            group_label = group_map.get(top_segment, top_segment)
            subgroup_label = subgroup_map.get(sub_segment, sub_segment)

            groups.setdefault(group_label, {"subgroups": {}})
            groups[group_label]["subgroups"].setdefault(
                subgroup_label, {"components": []}
            )
            groups[group_label]["subgroups"][subgroup_label]["components"].append(
                comp_id
            )

        # Pre-sort component lists and cache per-group counts so the render
        # loop never has to sort or count on re-runs.
        total_count = 0
        for g_data in groups.values():
            g_count = 0
            for sg_data in g_data["subgroups"].values():
                sg_data["components"].sort()
                g_count += len(sg_data["components"])
            g_data["count"] = g_count
            total_count += g_count

        tree[short_name] = {
            "file_path": fp,
            "meta": meta,
            "groups": groups,
            "count": total_count,
        }

    return tree


def _count_components(groups: Dict[str, dict]) -> int:
    """Return the total component count across all groups and sub-groups."""
    return sum(
        len(sg["components"])
        for g in groups.values()
        for sg in g["subgroups"].values()
    )


# ─── Leaf fragments ────────────────────────────────────────────────────────────
#
# Why fragments?
# ~~~~~~~~~~~~~~
# Previously the "Load details" button used the pattern:
#
#     if st.button(...):
#         st.session_state[load_key] = True
#         st.rerun()
#
# That click triggered two full script re-runs (one for the button event,
# one for the explicit st.rerun) and each re-run re-evaluated every open
# expander in the tree, producing a visible 1–2 s stutter.
#
# The fragment-based pattern below:
#   * sets the load flag and falls through to render the detail in the
#     SAME script execution (no explicit st.rerun), and
#   * scopes any subsequent reruns triggered from inside the leaf to the
#     fragment itself, so the rest of the tree is not re-evaluated.
#
# Requires Streamlit ≥ 1.37 for st.fragment. If you're pinned to an older
# version, drop the decorator — the single-rerun improvement still applies.

@st.fragment
def _render_seismic_leaf_fragment(
    *,
    comp_id: str,
    comp_data: dict,
    fp: str,
    fname: str,
    load_key: str,
    btn_key: str,
) -> None:
    """
    Render a seismic component leaf inside an isolated fragment.

    A "Load details" click sets the session-state flag and immediately
    falls through to render the detail panel in the same script
    execution, avoiding the double-rerun stutter of the previous pattern.
    """
    loaded = st.session_state.get(load_key, False)

    if not loaded:
        if st.button("Load details", key=btn_key, type="secondary"):
            st.session_state[load_key] = True
            loaded = True

    if loaded:
        if comp_data:
            render_component_leaf(comp_id, comp_data, fp)
        else:
            st.warning(
                f"Full data for `{comp_id}` was not found in `{fname}`. "
                "The component description is available but detailed fields "
                "are missing.",
                icon="⚠️",
            )


@st.fragment
def _render_wind_leaf_fragment(
    *,
    comp_id: str,
    comp_data: dict,
    fp: str,
    fname: str,
    load_key: str,
    btn_key: str,
) -> None:
    """Wind counterpart to ``_render_seismic_leaf_fragment``."""
    loaded = st.session_state.get(load_key, False)

    if not loaded:
        if st.button("Load details", key=btn_key, type="secondary"):
            st.session_state[load_key] = True
            loaded = True

    if loaded:
        if comp_data:
            render_wind_component_leaf(comp_id, comp_data, fp)
        else:
            st.warning(
                f"Full data for `{comp_id}` was not found in `{fname}`.",
                icon="⚠️",
            )


# ─── Tree renderer ─────────────────────────────────────────────────────────────

def render_seismic_tree(
    seismic_objects: Optional[List[SearchObject]] = None,
) -> None:
    """
    Render the seismic component library as a four-level collapsible tree.

    Parameters
    ----------
    seismic_objects : list of SearchObject, optional
        Pre-filtered list of seismic SearchObjects.  When None, all seismic
        objects are loaded from the cached FuzzyIndex.

    Performance strategy
    --------------------
    * The FuzzyIndex, _build_tree result, and all Plotly figures are cached
      at the process level — they survive re-runs without re-computation.
    * Streamlit executes expander child code on every re-run even when
      collapsed, so component detail panels (_render_component_detail) are
      guarded by a session-state set. Detail content is only rendered for
      components the user has explicitly opened, keeping the widget tree
      small on every run.
    """
    # ── Session state for tracking which components are open ───────────────
    if _EXPANDED_KEY not in st.session_state:
        st.session_state[_EXPANDED_KEY] = set()

    # ── Load data ──────────────────────────────────────────────────────────
    if seismic_objects is None:
        with st.spinner("Loading seismic fragility index…", show_time=True):
            seismic_objects = _get_cached_index().filter_by_hazard("seismic")

    if not seismic_objects:
        st.warning(
            "No seismic fragility data found. Check directory structure.",
            icon="⚠️",
        )
        return

    # Deduplicate file paths and pass a hashable tuple to the cached builder
    file_paths: tuple[str, ...] = tuple(
        dict.fromkeys(obj.file_path for obj in seismic_objects if obj.file_path)
    )
    tree = _build_tree(file_paths)

    # Build a file_path → SearchObject lookup to avoid O(N) scans per source.
    obj_by_path: Dict[str, SearchObject] = {
        o.file_path: o for o in seismic_objects if o.file_path
    }

    # Counts are stored in the tree dict by _build_tree — no re-computation needed.
    total = sum(src["count"] for src in tree.values())

    # ══ Root header ══════════════════════════════════════════════════════════
    st.markdown("## 🌍 Seismic")
    st.caption(
        f"{len(tree)} source{'s' if len(tree) != 1 else ''} · {total:,} components"
    )
    st.divider()

    for short_name, source_data in tree.items():
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]
        groups: Dict[str, dict] = source_data["groups"]

        obj = obj_by_path.get(fp)
        badge = _CATEGORY_BADGE.get(getattr(obj, "category", ""), "")
        n_comp = source_data["count"]

        # ══ Level 2: Source ══════════════════════════════════════════════════
        with st.expander(
            f"**{short_name}**  ·  {badge}  ·  `{n_comp:,}` components",
            expanded=False,
        ):
            if meta.get("Description"):
                st.caption(meta["Description"])

            version = meta.get("Version", "")
            fname = Path(fp).name if fp else "unknown"
            st.caption(f"Version: {version}  ·  File: `{fname}`")
            st.divider()

            for group_name, group_data in groups.items():
                group_total = group_data["count"]
                if group_total == 0:
                    continue

                # ══ Level 3: Component group ══════════════════════════════════
                with st.expander(
                    f"**{group_name}**  ·  `{group_total}` components",
                    expanded=False,
                ):
                    for sg_name, sg_data in group_data["subgroups"].items():
                        comps: List[str] = sg_data["components"]  # pre-sorted in _build_tree
                        if not comps:
                            continue

                        n_sg = len(comps)
                        sg_label = (
                            f"**{sg_name}**  ·  "
                            f"`{n_sg}` component{'s' if n_sg != 1 else ''}"
                        )

                        # ══ Level 4: Sub-group ════════════════════════════════
                        with st.expander(sg_label, expanded=False):
                            for comp_id in comps:
                                # comps is pre-sorted by _build_tree; no sort needed.
                                # _load_full_json is cached — O(1) after first call.
                                full_json: dict = _load_full_json(fp)
                                comp_data: dict = full_json.get(comp_id, {})
                                raw_desc: str = comp_data.get(
                                    "Description",
                                    obj.search_dict.get(comp_id, "") if obj else "",
                                )
                                preview = (
                                    raw_desc[:90] + "…"
                                    if len(raw_desc) > 90
                                    else raw_desc
                                )

                                # ══ Level 5: Component leaf ════════════════════
                                # Detail content is gated by a session-state
                                # flag so closed expanders don't build Plotly
                                # figures on every re-run. The leaf body lives
                                # inside an st.fragment so a click only reruns
                                # the leaf — not the entire tree.
                                load_key = f"loaded_{comp_id}"
                                btn_key = f"btn_{comp_id}"

                                with st.expander(
                                    f"🔩  **{comp_id}**  ·  {preview}",
                                    expanded=False,
                                ):
                                    _render_seismic_leaf_fragment(
                                        comp_id=comp_id,
                                        comp_data=comp_data,
                                        fp=fp,
                                        fname=fname,
                                        load_key=load_key,
                                        btn_key=btn_key,
                                    )


# ─── Wind tree renderer ────────────────────────────────────────────────────────

# Map top-level component prefixes to human-readable group labels.
# Derived from the IDs present in the SimCenter Wind Component Library.
_WIND_GROUP_LABELS: Dict[str, str] = {
    "DOOR":  "DOOR — Doors",
    "RCOV":  "RCOV — Roof Cover",
    "RSH":   "RSH — Roof Sheathing",
    "RWC":   "RWC — Roof-Wall Connections",
    "WALL":  "WALL — Walls",
    "WCOV":  "WCOV — Wall Cover",
    "WIN":   "WIN — Windows",
    "WSH":   "WSH — Wall Sheathing",
}

def render_wind_tree(
    wind_objects: Optional[List[SearchObject]] = None,
) -> None:
    """
    Render the SimCenter Wind Component Library as a collapsible tree.

    The tree has the same four-level structure used by ``render_seismic_tree``:

      Wind (Hurricane)
      └── Source  (SimCenter Wind Component Library)
          └── Component Group  (DOOR / WIN / RSH …)
              └── Sub-Group  (DOOR.garage / WIN.regular …)
                  └── Component  [detail panel + fragility chart]

    Parameters
    ----------
    wind_objects : list of SearchObject, optional
        Pre-filtered list of wind/hurricane component SearchObjects.
        When ``None``, all hurricane component objects are loaded from
        the cached FuzzyIndex, filtered to the
        ``hurricane/building/component`` path prefix so that Hazus
        *portfolio* sources are excluded.

    Performance strategy
    --------------------
    Identical to ``render_seismic_tree``: FuzzyIndex and _build_tree are
    cached at the process level; component detail panels are guarded by
    session-state flags so they are only rendered after an explicit
    "Load details" click.
    """
    # ── Session state ──────────────────────────────────────────────────────
    if _EXPANDED_KEY not in st.session_state:
        st.session_state[_EXPANDED_KEY] = set()

    # ── Load data ──────────────────────────────────────────────────────────
    if wind_objects is None:
        with st.spinner("Loading wind fragility index…"):
            all_hurricane = _get_cached_index().filter_by_hazard("hurricane")
            # Restrict to component-level sources only (exclude portfolio models
            # such as Hazus v5.1 which live under hurricane/building/portfolio).
            wind_objects = [
                o for o in all_hurricane
                if "/building/component/" in o.file_path
            ]

    if not wind_objects:
        st.warning(
            "No wind component fragility data found. "
            "Check that hurricane/building/component/ exists in the directory structure.",
            icon="⚠️",
        )
        return

    file_paths: tuple[str, ...] = tuple(
        dict.fromkeys(obj.file_path for obj in wind_objects if obj.file_path)
    )
    tree = _build_tree(file_paths)

    obj_by_path: Dict[str, SearchObject] = {
        o.file_path: o for o in wind_objects if o.file_path
    }

    total = sum(src["count"] for src in tree.values())

    # ══ Root header ══════════════════════════════════════════════════════════
    st.markdown("## 🌀 Wind (Hurricane)")
    st.caption(
        f"{len(tree)} source{'s' if len(tree) != 1 else ''} · {total:,} components"
    )
    st.divider()

    for short_name, source_data in tree.items():
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]
        groups: Dict[str, dict] = source_data["groups"]

        n_comp = source_data["count"]
        fname = Path(fp).name if fp else "unknown"

        # ══ Level 2: Source ══════════════════════════════════════════════════
        with st.expander(
            f"**{short_name}**  ·  🌐 SimCenter  ·  `{n_comp:,}` components",
            expanded=False,
        ):
            if meta.get("Description"):
                st.caption(meta["Description"])

            version = meta.get("Version", "")
            st.caption(f"Version: {version}  ·  File: `{fname}`")
            st.divider()

            for group_prefix, group_data in groups.items():
                group_total = group_data["count"]
                if group_total == 0:
                    continue

                # Apply human-readable label if available.
                group_label = _WIND_GROUP_LABELS.get(group_prefix, group_prefix)

                # ══ Level 3: Component group ══════════════════════════════════
                with st.expander(
                    f"**{group_label}**  ·  `{group_total}` components",
                    expanded=False,
                ):
                    for sg_name, sg_data in group_data["subgroups"].items():
                        comps: List[str] = sg_data["components"]
                        if not comps:
                            continue

                        n_sg = len(comps)
                        sg_label = (
                            f"**{sg_name}**  ·  "
                            f"`{n_sg}` component{'s' if n_sg != 1 else ''}"
                        )

                        # ══ Level 4: Sub-group ════════════════════════════════
                        with st.expander(sg_label, expanded=False):
                            for comp_id in comps:
                                full_json: dict = _load_full_json(fp)
                                comp_data_entry: dict = full_json.get(comp_id, {})
                                raw_desc: str = comp_data_entry.get("Description", "")
                                preview = (
                                    raw_desc[:90] + "…"
                                    if len(raw_desc) > 90
                                    else raw_desc
                                )

                                # ══ Level 5: Component leaf ════════════════════
                                load_key = f"wind_loaded_{comp_id}"
                                btn_key = f"wind_btn_{comp_id}"

                                with st.expander(
                                    f"🔩  **{comp_id}**  ·  {preview}",
                                    expanded=False,
                                ):
                                    _render_wind_leaf_fragment(
                                        comp_id=comp_id,
                                        comp_data=comp_data_entry,
                                        fp=fp,
                                        fname=fname,
                                        load_key=load_key,
                                        btn_key=btn_key,
                                    )