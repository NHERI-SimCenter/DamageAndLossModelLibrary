"""
tree_visuals.py
---------------
Hierarchical tree view for the fragility component library.

Renders a collapsible tree whose grouping depth mirrors each source's
``_GeneralInformation.ComponentGroups`` (FEMA P-58 nests three levels, Hazus two):

  Seismic
  └── Source                (FEMA P-58 / Hazus …)
      └── Group              (B - Shell …)
          └── Sub-Group      (B.10 - Super Structure …)
              └── Sub-Sub-Group  (B.10.31 - Steel Columns …)
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
from st_core.component import (
    _render_wind_component_detail,
    render_component_leaf,
    render_consequence_leaf,
    render_wind_component_leaf,
)

from st_visuals.helpers_visual import load_consequence_df



# ─── Palette & constants ───────────────────────────────────────────────────────

_CATEGORY_BADGE: Dict[str, str] = {"FEMA": "FEMA P-58", "HAZUS": "Hazus"}

# Top-level keys in a fragility.json that are not components.
_NON_COMPONENT_KEYS = {"References"}

# Session-state key prefix for "this sub-group has been opened at least once".
# Used to gate the inner per-component render loop so that closed sub-groups
# never iterate their (often hundreds of) components or instantiate Level-5
# expanders + fragments on every Streamlit re-run.
_OPENED_SUBGROUPS_KEY = "tree_opened_subgroups"






# ─── Data helpers ──────────────────────────────────────────────────────────────

# Filenames for each browsable dataset.
_DATASET_FILENAME = {
    "fragility": "fragility.json",
    "consequence": "consequence_repair.json",
}


@st.cache_data(show_spinner=False)
def _hazard_files(hazard: str, dataset: str = "fragility") -> tuple[str, ...]:
    """
    Tree-visible data files for a hazard + dataset, in stable order.

    Built from the same directory scope the search index uses
    (``tree_corpus_files``) so the tree and search never disagree, matched by
    ``Path.parts`` so it works regardless of OS path separators. For the
    consequence dataset the sibling ``consequence_repair.json`` is used, and
    sources that lack one (e.g. the SimCenter wind library) are dropped.
    """
    filename = _DATASET_FILENAME[dataset]
    files: list[str] = []
    for fp in tree_corpus_files("."):
        if hazard not in Path(fp).parts:
            continue
        candidate = Path(fp).with_name(filename)
        if candidate.exists():
            files.append(str(candidate))
    return tuple(files)


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

    The search index only stores component descriptions; this function
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
    Build a nested source → group hierarchy mirroring each file's
    ``_GeneralInformation.ComponentGroups``.

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
                    "components": [(comp_id, preview), ...]
                  }
                }
              }
            }
          }
        }

        ``preview`` is the truncated component description used as the
        Level-5 expander header label.  It is computed here, once, so that
        the render loop never has to touch ``fragility.json`` to display
        a collapsed leaf header.

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

        meta: dict = data.get("_GeneralInformation", {}) or {}
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

        _sort_node(root)

        # Route each component into group -> subgroup buckets.
        # Top-level group: first dot-segment of comp_id (e.g. "GF", "STR").
        # Sub-group: first two dot-segments joined (e.g. "GF.H", "STR.W1").
        # When a prefix has no match in the maps (happens for FEMA P-58 IDs
        # whose sub-groups may not be listed), fall back to the bare segment.
        #
        # The description preview used for the Level-5 expander header is
        # computed here, once, while we already have ``data`` loaded.  This
        # removes the per-component _load_full_json() / dict.get() /
        # string-slice work that previously ran on every Streamlit re-run.
        # Components are stored as ``(comp_id, preview)`` tuples.
        groups: Dict[str, dict] = {}
        for comp_id in comp_ids:
            parts = comp_id.split(".")
            top_segment = parts[0]
            sub_segment = ".".join(parts[:2]) if len(parts) >= 2 else top_segment

            group_label = group_map.get(top_segment, top_segment)
            subgroup_label = subgroup_map.get(sub_segment, sub_segment)

            raw_desc = data.get(comp_id, {}).get("Description", "") or ""
            preview = raw_desc[:90] + "…" if len(raw_desc) > 90 else raw_desc

            groups.setdefault(group_label, {"subgroups": {}})
            groups[group_label]["subgroups"].setdefault(
                subgroup_label, {"components": []}
            )
            groups[group_label]["subgroups"][subgroup_label]["components"].append(
                (comp_id, preview)
            )

        # Pre-sort component lists and cache per-group counts so the render
        # loop never has to sort or count on re-runs.  Sort by comp_id (the
        # first tuple element) so the ordering matches the previous behaviour.
        total_count = 0
        for g_data in groups.values():
            g_count = 0
            for sg_data in g_data["subgroups"].values():
                sg_data["components"].sort(key=lambda t: t[0])
                g_count += len(sg_data["components"])
            g_data["count"] = g_count
            total_count += g_count

        tree[short_name] = {
            "file_path": fp,
            "short_name": short_name,
            "meta": meta,
            "root": root,
        }

    return tree


def _make_node(label: str, prefix: str) -> dict:
    return {"label": label, "prefix": prefix, "children": {}, "components": []}


def _build_skeleton(cg, parent: dict, node_by_prefix: Dict[str, dict]) -> None:
    """
    Recursively turn a ComponentGroups structure into grouping nodes.

    Handles the nested ``{label: {label: [label, …]}}`` form (FEMA, wind) and
    the flatter ``{label: [...]}`` / ``[label, …]`` forms (Hazus). Every label is
    ``"<prefix> - <name>"``; the prefix is the routing key, the whole string the
    display label.
    """
    if isinstance(cg, dict):
        items = list(cg.items())
    elif isinstance(cg, list):
        items = [(label, None) for label in cg]
    else:
        return

    for label, child in items:
        if not isinstance(label, str):
            continue
        prefix = label.split(" - ", 1)[0].strip()
        node = node_by_prefix.get(prefix)
        if node is None:
            node = _make_node(label, prefix)
            parent["children"][prefix] = node
            node_by_prefix[prefix] = node
        if child is not None:
            _build_skeleton(child, node, node_by_prefix)


def _route_component(comp_id: str, node_by_prefix: Dict[str, dict], root: dict) -> dict:
    """Return the deepest group node whose prefix is a prefix of *comp_id*."""
    segments = comp_id.split(".")
    # Longest prefix first (drop the trailing component-number segment(s)).
    for k in range(len(segments) - 1, 0, -1):
        node = node_by_prefix.get(".".join(segments[:k]))
        if node is not None:
            return node
    # No ComponentGroups prefix matched — bucket under the bare first segment.
    seg0 = segments[0]
    node = node_by_prefix.get(seg0)
    if node is None:
        node = _make_node(seg0, seg0)
        root["children"][seg0] = node
        node_by_prefix[seg0] = node
    return node


def _sort_node(node: dict) -> None:
    """Sort a node's loose components (children keep ComponentGroups order)."""
    node["components"].sort()
    for child in node["children"].values():
        _sort_node(child)


def _prune_node(node: dict, allowed_ids: Optional[set]) -> tuple[Optional[dict], int]:
    """
    Filter a node subtree to ``allowed_ids``, dropping empties.

    Returns ``(pruned_node, count)``, or ``(None, 0)`` when nothing survives.
    With ``allowed_ids is None`` everything is kept; counts are computed either
    way so labels stay correct under filtering.
    """
    children: Dict[str, dict] = {}
    total = 0
    for prefix, child in node["children"].items():
        pruned, cnt = _prune_node(child, allowed_ids)
        if pruned is not None:
            children[prefix] = pruned
            total += cnt

    comps = (
        node["components"]
        if allowed_ids is None
        else [c for c in node["components"] if c in allowed_ids]
    )
    total += len(comps)

    if total == 0:
        return None, 0

    return {
        "label": node["label"],
        "prefix": node["prefix"],
        "children": children,
        "components": comps,
        "count": total,
    }, total


def _build_render_plan(
    tree: Dict[str, dict],
    allowed_ids: Optional[set],
) -> tuple[list, int]:
    """
    Build a per-source render plan, pruned to ``allowed_ids``.

    Returns ``(plan, grand_total)`` where each plan entry is
    ``(short_name, source_data, pruned_root, count)``. Sources with no surviving
    components are dropped.
    """
    plan: list = []
    grand_total = 0

    for source_data in tree.values():
        pruned_root, count = _prune_node(source_data["root"], allowed_ids)
        if pruned_root is not None:
            plan.append((source_data["short_name"], source_data, pruned_root, count))
            grand_total += count

    return plan, grand_total


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

def _render_leaf(
    comp_id: str, fp: str, hazard: str, dataset: str, key_prefix: str
) -> None:
    """
    Render a single leaf with the lazy "Load details" guard.

    The expander body runs on every re-run regardless of open state, so detail
    content (tabs / Plotly charts) is gated behind a button to keep re-runs
    cheap. ``key_prefix`` is unique per source, so the same component ID
    appearing in two sources (e.g. Hazus v5.1 and v6.1) never collides on keys.
    The leaf renderer is chosen by ``dataset`` (consequence) and ``hazard``.
    """
    # ── Session state for tracking which components are open ───────────────
    if _EXPANDED_KEY not in st.session_state:
        st.session_state[_EXPANDED_KEY] = set()
    if _OPENED_SUBGROUPS_KEY not in st.session_state:
        st.session_state[_OPENED_SUBGROUPS_KEY] = set()

    # ── Load data ──────────────────────────────────────────────────────────
    if seismic_objects is None:
        with st.spinner("Loading seismic fragility index…", show_time=True):
            seismic_objects = _get_cached_index().filter_by_hazard("seismic")


def _render_node(
    node: dict, fp: str, hazard: str, dataset: str, filtering: bool, key_prefix: str
) -> None:
    """
    Recursively render a grouping node: child groups first, then the component
    leaves held directly at this level.

    In browse mode the leaf rows are deferred behind a "Show N components"
    checkbox, so a closed branch builds a single widget instead of hundreds of
    leaf expanders on every re-run (Streamlit executes expander bodies even
    while collapsed). In a filtered view the matches are few and meant to be
    seen, so they render immediately.
    """
    for child in node["children"].values():
        with st.expander(
            f"**{child['label']}**  ·  `{child['count']:,}` components",
            expanded=filtering,
        ):
            _render_node(child, fp, hazard, dataset, filtering, key_prefix)

    comps = node["components"]
    if not comps:
        return

    # `filtering or st.checkbox(...)` short-circuits so the checkbox is only
    # built (and the leaves only deferred) in browse mode.
    if filtering or st.checkbox(
        f"Show {len(comps)} component{'s' if len(comps) != 1 else ''}",
        key=f"{key_prefix}show_{node['prefix']}",
    ):
        for comp_id in comps:
            _render_leaf(comp_id, fp, hazard, dataset, key_prefix)


def _render_tree(
    file_paths: tuple[str, ...],
    *,
    hazard: str,
    dataset: str,
    header: str,
    no_data_warning: str,
    no_match_info: str,
    allowed_ids: Optional[set],
) -> None:
    """Shared renderer for the fragility and consequence trees (depth follows the data)."""
    if not file_paths:
        st.warning(no_data_warning, icon="⚠️")
        return

    tree = _build_tree(file_paths)

    # Stable per-source key prefix (independent of filtering) so widget keys
    # stay unique even when two sources share a component ID — and across trees
    # (dataset + hazard), since fragility and consequence trees can render on the
    # same page ("All" browse) and share component IDs (e.g. FEMA B.10.31).
    src_prefix = {fp: f"{dataset}_{hazard}_{i}_" for i, fp in enumerate(tree)}

    # Prune to allowed_ids (no-op when None). A filtered view auto-expands so
    # matches are visible without clicking through every level.
    plan, total = _build_render_plan(tree, allowed_ids)
    filtering = allowed_ids is not None

    st.markdown(header)
    st.caption(
        f"{len(plan)} source{'s' if len(plan) != 1 else ''} · {total:,} components"
    )
    st.divider()

    if not plan:
        st.info(no_match_info, icon="🔍")
        return

    for short_name, source_data, root, n_comp in plan:
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]
        badge = (
            "🌐 SimCenter"
            if hazard == "hurricane"
            else _CATEGORY_BADGE.get(_category_of(fp), "")
        )

        # ══ Source ════════════════════════════════════════════════════════════
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

            _render_node(root, fp, hazard, dataset, filtering, src_prefix[fp])

                # ══ Level 3: Component group ══════════════════════════════════
                with st.expander(
                    f"**{group_name}**  ·  `{group_total}` components",
                    expanded=False,
                ):
                    for sg_name, sg_data in group_data["subgroups"].items():
                        comps: List[tuple[str, str]] = sg_data["components"]  # pre-sorted (id, preview)
                        if not comps:
                            continue

def render_seismic_tree(
    seismic_objects: Optional[list] = None,
    allowed_ids: Optional[set] = None,
) -> None:
    """
    Render the seismic component library as a collapsible tree.

                        # ══ Level 4: Sub-group ════════════════════════════════
                        # Streamlit re-runs all expander children regardless of
                        # open/closed state.  Without a gate at this level, a
                        # re-run would iterate every component in every sub-group
                        # on the page and instantiate a Level-5 st.expander +
                        # st.fragment + st.button for each one — thousands of
                        # widgets even when nothing visible is expanded.
                        #
                        # The "Show components" button below records that the
                        # user has opened this sub-group at least once.  After
                        # that, the inner loop runs (the user's collapsing the
                        # outer expander doesn't reset the flag — but the leaf
                        # widgets only do real work after their own "Load
                        # details" click, so the only residual cost is creating
                        # the Level-5 expander shells, which is cheap).
                        sg_key = f"seismic::{short_name}::{group_name}::{sg_name}"

                        with st.expander(sg_label, expanded=False):
                            opened = sg_key in st.session_state[_OPENED_SUBGROUPS_KEY]

                            if not opened:
                                if st.button(
                                    f"Show {n_sg} component{'s' if n_sg != 1 else ''}",
                                    key=f"sg_btn_{sg_key}",
                                    type="secondary",
                                ):
                                    st.session_state[_OPENED_SUBGROUPS_KEY].add(sg_key)
                                    opened = True

                            if opened:
                                # Hoist the source JSON load out of the inner
                                # loop — it is invariant across all components
                                # in this sub-group.  Cached, so this is also
                                # O(1) after the first call per process.
                                full_json: dict = _load_full_json(fp)

                                for comp_id, preview in comps:
                                    comp_data: dict = full_json.get(comp_id, {})

                                    # ══ Level 5: Component leaf ════════════════
                                    # Detail content is gated by a session-state
                                    # flag so closed expanders don't build Plotly
                                    # figures on every re-run.  The leaf body
                                    # lives inside an st.fragment so a click only
                                    # reruns the leaf — not the entire tree.
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
    wind_objects: Optional[list] = None,
    allowed_ids: Optional[set] = None,
) -> None:
    """
    Render the SimCenter Wind Component Library as a collapsible tree.

    Same structure as ``render_seismic_tree``, driven by the wind library's
    ``ComponentGroups`` (e.g. ``RWC - Roof-Wall Connection → RWC.toe_nail - Toe
    nail → RWC.toe_nail.straps - Toe nail with straps → components``).

    Parameters
    ----------
    wind_objects : list, optional
        Pre-filtered list of objects exposing a ``file_path`` attribute. When
        ``None``, the shared tree-file list supplies the hurricane component
        libraries (``hurricane/building/component/``); Hazus portfolio sources
        are excluded by construction.
    allowed_ids : set of str, optional
        When provided, only components whose ID is in this set are shown, and
        empty branches are hidden.
    """
    # ── Session state ──────────────────────────────────────────────────────
    if _EXPANDED_KEY not in st.session_state:
        st.session_state[_EXPANDED_KEY] = set()
    if _OPENED_SUBGROUPS_KEY not in st.session_state:
        st.session_state[_OPENED_SUBGROUPS_KEY] = set()

    # ── Load data ──────────────────────────────────────────────────────────
    if wind_objects is None:
        file_paths = _hazard_files("hurricane")
    else:
        file_paths = tuple(
            dict.fromkeys(
                o.file_path for o in wind_objects if getattr(o, "file_path", "")
            )
        )

    _render_tree(
        file_paths,
        hazard="hurricane",
        dataset="fragility",
        header="## 🌀 Wind (Hurricane)",
        no_data_warning=(
            "No wind component fragility data found. "
            "Check that hurricane/building/component/ exists in the directory structure."
        ),
        no_match_info="No wind components match the current filters.",
        allowed_ids=allowed_ids,
    )


def render_consequence_tree(allowed_ids: Optional[set] = None) -> None:
    """
    Render the repair-consequence library as a collapsible tree.

    Browsed via the dataset toggle. Unlike fragility, consequence records are
    keyed by component (FEMA P-58) or by occupancy class (Hazus — e.g.
    ``STR - Structural → STR.RES1 Single-family Dwelling``), so this is a
    separate tree: it surfaces consequence records that have no fragility
    component and therefore never appear in the fragility tree. The SimCenter
    wind library carries no consequence data, so only seismic sources appear.

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
                        comps: List[tuple[str, str]] = sg_data["components"]  # (id, preview)
                        if not comps:
                            continue

                        n_sg = len(comps)
                        sg_label = (
                            f"**{sg_name}**  ·  "
                            f"`{n_sg}` component{'s' if n_sg != 1 else ''}"
                        )

                        # ══ Level 4: Sub-group ════════════════════════════════
                        # See render_seismic_tree for the rationale — same gate.
                        sg_key = f"wind::{short_name}::{group_prefix}::{sg_name}"

                        with st.expander(sg_label, expanded=False):
                            opened = sg_key in st.session_state[_OPENED_SUBGROUPS_KEY]

                            if not opened:
                                if st.button(
                                    f"Show {n_sg} component{'s' if n_sg != 1 else ''}",
                                    key=f"sg_btn_{sg_key}",
                                    type="secondary",
                                ):
                                    st.session_state[_OPENED_SUBGROUPS_KEY].add(sg_key)
                                    opened = True

                            if opened:
                                full_json: dict = _load_full_json(fp)

                                for comp_id, preview in comps:
                                    comp_data_entry: dict = full_json.get(comp_id, {})

                                    # ══ Level 5: Component leaf ════════════════
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
