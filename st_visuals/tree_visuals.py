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

from st_search.semantic_index import tree_corpus_files
from st_core.component import _render_wind_component_detail, render_component_leaf, render_wind_component_leaf

from st_visuals.helpers_visual import load_consequence_df



# ─── Palette & constants ───────────────────────────────────────────────────────

_CATEGORY_BADGE: Dict[str, str] = {"FEMA": "🔵 FEMA P-58", "HAZUS": "🟠 Hazus"}

# Session-state key that holds the set of expanded component IDs
_EXPANDED_KEY = "tree_expanded_components"






# ─── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _hazard_files(hazard: str) -> tuple[str, ...]:
    """
    Tree-visible fragility.json paths for a hazard, in stable order.

    Sourced from the same corpus the search index uses (``tree_corpus_files``),
    so the tree and search never disagree — and matched by ``Path.parts`` so it
    works regardless of OS path separators.
    """
    return tuple(fp for fp in tree_corpus_files(".") if hazard in Path(fp).parts)


def _category_of(file_path: str) -> str:
    """FEMA / HAZUS badge category parsed from the source path."""
    if "FEMA" in file_path:
        return "FEMA"
    if "Hazus" in file_path:
        return "HAZUS"
    return ""


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

        # Build prefix → label maps from ComponentGroups.
        #
        # ComponentGroups is a dict[str, list[str]]:
        #   { "GF": ["GF.H", "GF.V", "GF.L"],
        #     "STR": ["STR.W1", "STR.S1", ...], ... }
        #
        # Keys are top-level group prefixes; values are lists of subgroup
        # prefixes.  We build two independent maps:
        #   group_map:    top-prefix  → group display label  (e.g. "GF")
        #   subgroup_map: sub-prefix  → subgroup display label (e.g. "GF.H")
        # Both maps store just the prefix as the label because the JSON does
        # not separately provide human-readable group names here.
        raw_cg = meta.get("ComponentGroups", {})
        if not isinstance(raw_cg, dict):
            raw_cg = {}

        group_map: Dict[str, str] = {grp: grp for grp in raw_cg}
        subgroup_map: Dict[str, str] = {
            sg: sg
            for sg_list in raw_cg.values()
            if isinstance(sg_list, list)
            for sg in sg_list
        }

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


def _build_render_plan(
    tree: Dict[str, dict],
    allowed_ids: Optional[set],
) -> tuple[list, int]:
    """
    Flatten a built tree into a render plan, pruned to ``allowed_ids``.

    When ``allowed_ids`` is None the full tree is returned. Otherwise only
    components in the set survive, and any sub-group / group / source that ends
    up empty is dropped — so a filtered tree shows just the matching branches
    with correct counts.

    Returns
    -------
    (plan, grand_total)
        ``plan`` is a list of per-source tuples::

            (short_name, source_data, [(group_name, [(sg_name, [comp_id, …])], group_total)], source_total)
    """
    plan: list = []
    grand_total = 0

    for short_name, source_data in tree.items():
        groups_plan: list = []
        source_total = 0

        for group_name, group_data in source_data["groups"].items():
            sg_plan: list = []
            group_total = 0

            for sg_name, sg_data in group_data["subgroups"].items():
                comps: List[str] = sg_data["components"]
                visible = (
                    comps
                    if allowed_ids is None
                    else [c for c in comps if c in allowed_ids]
                )
                if visible:
                    sg_plan.append((sg_name, visible))
                    group_total += len(visible)

            if sg_plan:
                groups_plan.append((group_name, sg_plan, group_total))
                source_total += group_total

        if source_total:
            plan.append((short_name, source_data, groups_plan, source_total))
            grand_total += source_total

    return plan, grand_total


# ─── Tree renderer ─────────────────────────────────────────────────────────────

def render_seismic_tree(
    seismic_objects: Optional[list] = None,
    allowed_ids: Optional[set] = None,
) -> None:
    """
    Render the seismic component library as a four-level collapsible tree.

    Parameters
    ----------
    seismic_objects : list, optional
        Pre-filtered list of objects exposing a ``file_path`` attribute. When
        None, the seismic corpus is loaded from the shared tree-file list.
    allowed_ids : set of str, optional
        When provided, only components whose ID is in this set are shown, and
        empty sub-groups / groups / sources are hidden. Used by the search panel
        to prune the tree to a facet selection. When None, the full tree renders.

    Performance strategy
    --------------------
    * The file list, _build_tree result, and all Plotly figures are cached
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
        file_paths = _hazard_files("seismic")
    else:
        file_paths = tuple(
            dict.fromkeys(
                o.file_path for o in seismic_objects if getattr(o, "file_path", "")
            )
        )

    if not file_paths:
        st.warning(
            "No seismic fragility data found. Check directory structure.",
            icon="⚠️",
        )
        return

    tree = _build_tree(file_paths)

    # Prune to allowed_ids (no-op when None). A filtered view auto-expands so
    # matches are visible without clicking through every level.
    plan, total = _build_render_plan(tree, allowed_ids)
    filtering = allowed_ids is not None

    # ══ Root header ══════════════════════════════════════════════════════════
    st.markdown("## 🌍 Seismic")
    st.caption(
        f"{len(plan)} source{'s' if len(plan) != 1 else ''} · {total:,} components"
    )
    st.divider()

    if not plan:
        st.info("No seismic components match the current filters.", icon="🔍")
        return

    for short_name, source_data, groups_plan, n_comp in plan:
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]

        badge = _CATEGORY_BADGE.get(_category_of(fp), "")

        # ══ Level 2: Source ══════════════════════════════════════════════════
        with st.expander(
            f"**{short_name}**  ·  {badge}  ·  `{n_comp:,}` components",
            expanded=filtering,
        ):
            if meta.get("Description"):
                st.caption(meta["Description"])

            version = meta.get("Version", "")
            fname = Path(fp).name if fp else "unknown"
            st.caption(f"Version: {version}  ·  File: `{fname}`")
            st.divider()

            for group_name, sg_plan, group_total in groups_plan:
                # ══ Level 3: Component group ══════════════════════════════════
                with st.expander(
                    f"**{group_name}**  ·  `{group_total}` components",
                    expanded=filtering,
                ):
                    for sg_name, comps in sg_plan:
                        n_sg = len(comps)
                        sg_label = (
                            f"**{sg_name}**  ·  "
                            f"`{n_sg}` component{'s' if n_sg != 1 else ''}"
                        )

                        # ══ Level 4: Sub-group ════════════════════════════════
                        with st.expander(sg_label, expanded=filtering):
                            for comp_id in comps:
                                # _load_full_json is cached — O(1) after first call.
                                full_json: dict = _load_full_json(fp)
                                comp_data: dict = full_json.get(comp_id, {})
                                raw_desc: str = comp_data.get("Description", "")
                                preview = (
                                    raw_desc[:90] + "…"
                                    if len(raw_desc) > 90
                                    else raw_desc
                                )

                                # ══ Level 5: Component leaf ════════════════════
                                # Session-state guard: detail content is only
                                # rendered after an explicit "Load" click.
                                # Without this guard, Streamlit executes every
                                # expander's body on every re-run (open or not),
                                # so st.tabs / st.plotly_chart / _render_consequence_tab
                                # would fire for ALL components on every interaction.
                                load_key = f"loaded_{comp_id}"
                                with st.expander(
                                    f"🔩  **{comp_id}**  ·  {preview}",
                                    expanded=False,
                                ):
                                    if load_key not in st.session_state:
                                        if st.button(
                                            "Load details",
                                            key=f"btn_{comp_id}",
                                            type="secondary",
                                        ):
                                            st.session_state[load_key] = True
                                            st.rerun()
                                    elif comp_data:
                                        render_component_leaf(
                                            comp_id, comp_data, fp
                                        )
                                    else:
                                        st.warning(
                                            f"Full data for `{comp_id}` was not found "
                                            f"in `{fname}`. The component description "
                                            "is available but detailed fields are missing.",
                                            icon="⚠️",
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
    wind_objects: Optional[list] = None,
    allowed_ids: Optional[set] = None,
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
    wind_objects : list, optional
        Pre-filtered list of objects exposing a ``file_path`` attribute. When
        ``None``, the shared tree-file list supplies the hurricane component
        libraries (``hurricane/building/component/``); Hazus *portfolio* sources
        are excluded by construction.
    allowed_ids : set of str, optional
        When provided, only components whose ID is in this set are shown, and
        empty branches are hidden (see ``render_seismic_tree``).

    Performance strategy
    --------------------
    Identical to ``render_seismic_tree``: the file list and _build_tree are
    cached at the process level; component detail panels are guarded by
    session-state flags so they are only rendered after an explicit
    "Load details" click.
    """
    # ── Session state ──────────────────────────────────────────────────────
    if _EXPANDED_KEY not in st.session_state:
        st.session_state[_EXPANDED_KEY] = set()

    # ── Load data ──────────────────────────────────────────────────────────
    # The shared corpus already restricts hurricane to hurricane/building/component/
    # so portfolio models (Hazus v5.1) are excluded.
    if wind_objects is None:
        file_paths = _hazard_files("hurricane")
    else:
        file_paths = tuple(
            dict.fromkeys(
                o.file_path for o in wind_objects if getattr(o, "file_path", "")
            )
        )

    if not file_paths:
        st.warning(
            "No wind component fragility data found. "
            "Check that hurricane/building/component/ exists in the directory structure.",
            icon="⚠️",
        )
        return

    tree = _build_tree(file_paths)

    plan, total = _build_render_plan(tree, allowed_ids)
    filtering = allowed_ids is not None

    # ══ Root header ══════════════════════════════════════════════════════════
    st.markdown("## 🌀 Wind (Hurricane)")
    st.caption(
        f"{len(plan)} source{'s' if len(plan) != 1 else ''} · {total:,} components"
    )
    st.divider()

    if not plan:
        st.info("No wind components match the current filters.", icon="🔍")
        return

    for short_name, source_data, groups_plan, n_comp in plan:
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]
        fname = Path(fp).name if fp else "unknown"

        # ══ Level 2: Source ══════════════════════════════════════════════════
        with st.expander(
            f"**{short_name}**  ·  🌐 SimCenter  ·  `{n_comp:,}` components",
            expanded=filtering,
        ):
            if meta.get("Description"):
                st.caption(meta["Description"])

            version = meta.get("Version", "")
            st.caption(f"Version: {version}  ·  File: `{fname}`")
            st.divider()

            for group_prefix, sg_plan, group_total in groups_plan:
                # Apply human-readable label if available.
                group_label = _WIND_GROUP_LABELS.get(group_prefix, group_prefix)

                # ══ Level 3: Component group ══════════════════════════════════
                with st.expander(
                    f"**{group_label}**  ·  `{group_total}` components",
                    expanded=filtering,
                ):
                    for sg_name, comps in sg_plan:
                        n_sg = len(comps)
                        sg_label = (
                            f"**{sg_name}**  ·  "
                            f"`{n_sg}` component{'s' if n_sg != 1 else ''}"
                        )

                        # ══ Level 4: Sub-group ════════════════════════════════
                        with st.expander(sg_label, expanded=filtering):
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
                                with st.expander(
                                    f"🔩  **{comp_id}**  ·  {preview}",
                                    expanded=False,
                                ):
                                    if load_key not in st.session_state:
                                        if st.button(
                                            "Load details",
                                            key=f"wind_btn_{comp_id}",
                                            type="secondary",
                                        ):
                                            st.session_state[load_key] = True
                                            st.rerun()
                                    elif comp_data_entry:
                                        render_wind_component_leaf(
                                            comp_id, comp_data_entry, fp
                                        )
                                    else:
                                        st.warning(
                                            f"Full data for `{comp_id}` was not found "
                                            f"in `{fname}`.",
                                            icon="⚠️",
                                        )