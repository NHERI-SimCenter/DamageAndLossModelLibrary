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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from st_search.component_search import FuzzyIndex, SearchObject


# ─── Palette & constants ───────────────────────────────────────────────────────

_DS_COLORS: List[str] = ["#3b82f6", "#f59e0b", "#ef4444", "#7c3aed", "#10b981"]
_CATEGORY_BADGE: Dict[str, str] = {"FEMA": "🔵 FEMA P-58", "HAZUS": "🟠 Hazus"}

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
            "meta": dict,
            "category": str,
            "short_name": str,
            "search_dict": {comp_id: description},
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
    """
    # Load each JSON and build SearchObject-like info without holding the
    # non-hashable SearchObject itself in the cached return value.
    from st_search.component_search import extract_search_metadata, SearchObject

    tree: Dict[str, dict] = {}

    for fp in file_paths:
        try:
            combined = extract_search_metadata(fp)
        except Exception:
            continue

        obj = SearchObject(combined_dict=combined, file_path=fp)
        short_name = obj.short_name

        # Read ComponentGroups directly from the raw JSON via json.load so the
        # original dict[str, list[str]] structure is always preserved.
        # extract_search_metadata routes _GeneralInformation through pandas
        # (.to_dict() on a Series), which can silently convert a dict-of-lists
        # into a plain list of keys — making subgroup_by_prefix empty and
        # routing every component into the Unclassified fallback bucket.
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                _raw = json.load(fh)
            _raw_groups = _raw.get("_GeneralInformation", {}).get("ComponentGroups", {})
        except Exception:
            _raw_groups = {}

        component_groups: Dict[str, List[str]] = {
            k: (v if isinstance(v, list) else [])
            for k, v in _raw_groups.items()
        } if isinstance(_raw_groups, dict) else {}

        # ── Enrich component_groups with auto-derived sub-groups ──────────
        # Some sources (e.g. FEMA P-58) store ComponentGroups with empty
        # sub-group lists: {"B - Shell": [], "C - Interiors": []}.
        # Classification is instead encoded in the component ID itself:
        #   B.10.31.001a  →  group "B", sub-group "B.10.31"
        # When sg_list is empty we derive sub-groups by grouping all
        # matching component IDs on their first 3 dot-separated segments.
        enriched_groups: Dict[str, List[str]] = {}
        for group_name, sg_list in component_groups.items():
            if sg_list:
                # Explicit sub-groups present (Hazus style) — use as-is
                enriched_groups[group_name] = sg_list
            else:
                # Derive sub-groups from component IDs (FEMA P-58 style)
                gpfx = group_name.split(" - ")[0].strip()
                derived: Dict[str, None] = {}  # ordered set
                for comp_id in obj.search_dict:
                    if comp_id.startswith(gpfx):
                        parts = comp_id.split(".")
                        # Take first 3 segments: B.10.31 (skip leaf variant)
                        sg_pfx = ".".join(parts[:3]) if len(parts) >= 3 else comp_id
                        derived[sg_pfx] = None
                enriched_groups[group_name] = list(derived.keys())

        # Build prefix look-ups (longest prefix wins during routing)
        subgroup_by_prefix: Dict[str, str] = {}
        subgroup_parent: Dict[str, str] = {}
        for group_name, sg_list in enriched_groups.items():
            for sg in sg_list:
                # For explicit sub-groups the prefix is before " - ";
                # for derived ones the entry IS already the prefix string.
                pfx = sg.split(" - ")[0].strip() if " - " in sg else sg
                subgroup_by_prefix[pfx] = sg
                subgroup_parent[sg] = group_name

        groups: Dict[str, dict] = {}
        for group_name, sg_list in enriched_groups.items():
            groups[group_name] = {
                "subgroups": {sg: {"components": []} for sg in sg_list}
            }

        # Route every component to its deepest matching sub-group
        for comp_id in obj.search_dict:
            placed = False
            for pfx in sorted(subgroup_by_prefix, key=len, reverse=True):
                if comp_id.startswith(pfx):
                    sg = subgroup_by_prefix[pfx]
                    g = subgroup_parent[sg]
                    groups[g]["subgroups"][sg]["components"].append(comp_id)
                    placed = True
                    break

            if not placed:
                # True fallback — component prefix matched no group at all
                for group_name in enriched_groups:
                    gpfx = group_name.split(" - ")[0].strip()
                    if comp_id.startswith(gpfx):
                        bucket = f"{gpfx} - (Unclassified)"
                        if bucket not in groups[group_name]["subgroups"]:
                            groups[group_name]["subgroups"][bucket] = {"components": []}
                        groups[group_name]["subgroups"][bucket]["components"].append(comp_id)
                        placed = True
                        break

        tree[short_name] = {
            "file_path": fp,
            "meta": obj.general_info_dict,
            "category": obj.category,
            "short_name": short_name,
            "search_dict": obj.search_dict,
            "groups": groups,
        }

    return tree


def _count_components(groups: Dict[str, dict]) -> int:
    return sum(
        len(sg["components"])
        for g in groups.values()
        for sg in g["subgroups"].values()
    )


# ─── Plotly helpers (cached per component ID) ─────────────────────────────────

@st.cache_data(show_spinner=False)
def _make_fragility_figure(comp_id: str, limit_states_json: str) -> go.Figure:
    """
    Build and cache a lognormal fragility figure.

    limit_states_json is the JSON-serialised LimitStates dict so the result
    is hashable for st.cache_data.
    """
    limit_states: dict = json.loads(limit_states_json)
    x = np.linspace(1e-4, 2.0, 300)
    fig = go.Figure()
    i = 0

    for ls_key, ls_data in limit_states.items():
        for ds_key, ds_data in ls_data.items():
            median = 0.15 * (i + 1)
            beta = 0.60
            with np.errstate(divide="ignore", invalid="ignore"):
                y = norm.cdf(np.log(np.maximum(x, 1e-9) / median) / beta)
            ds_label = (ds_data.get("Description", "") if isinstance(ds_data, dict) else str(ds_data))[:50]

            fig.add_trace(go.Scatter(
                x=x, y=y,
                name=f"{ds_key}: {ds_label}…",
                mode="lines",
                line=dict(color=_DS_COLORS[i % len(_DS_COLORS)], width=2.5),
                hovertemplate=(
                    f"<b>{ds_key}</b><br>"
                    "PGA: %{x:.3f} g<br>"
                    f"P(DS ≥ {ds_key}): %{{y:.1%}}<extra></extra>"
                ),
            ))
            i += 1

    fig.update_layout(
        xaxis_title="Peak Ground Acceleration (g)",
        yaxis=dict(title="P(DS ≥ ds | IM)", tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", y=-0.38, font=dict(size=10)),
        height=340,
        margin=dict(l=60, r=20, t=36, b=130),
        template="plotly_white",
        annotations=[dict(
            text="Illustrative lognormal curves — θ / β parameters pending integration",
            xref="paper", yref="paper", x=0.5, y=1.06,
            showarrow=False, font=dict(size=9, color="#9ca3af"), xanchor="center",
        )],
    )
    return fig


@st.cache_data(show_spinner=False)
def _make_consequence_placeholder() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="Consequence model parameters pending integration",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=13, color="#9ca3af"), xanchor="center",
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=300, template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ─── Component detail panel ────────────────────────────────────────────────────

def _render_component_detail(comp_id: str, comp_data: dict) -> None:
    """
    Render the inline detail panel for a leaf component node.

    Only called when the user has explicitly opened the component — never
    rendered unconditionally inside a collapsed expander.
    """
    description = comp_data.get("Description", "")
    comments = comp_data.get("Comments", "")
    block_size = comp_data.get("SuggestedComponentBlockSize", "")
    round_up = comp_data.get("RoundUpToIntegerQuantity", "")
    limit_states: dict = comp_data.get("LimitStates", {})

    col_left, col_right = st.columns([1, 2], gap="large")

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
                        if isinstance(ds_data, dict) else str(ds_data)
                    )
                    with st.expander(f"{ls_key} / {ds_key}", expanded=False):
                        st.caption(desc_text)
        else:
            st.caption("No limit-state data found.")

    with col_right:
        if comments:
            with st.expander("Technical notes / comments", expanded=False):
                st.caption(comments)

        tab_frag, tab_cons = st.tabs(["Fragility curves", "Consequence curves"])

        with tab_frag:
            if limit_states:
                st.plotly_chart(
                    _make_fragility_figure(comp_id, json.dumps(limit_states)),
                    use_container_width=True,
                    key=f"frag_{comp_id}",
                )
            else:
                st.info("No limit-state data available to generate curves.", icon="ℹ️")

        with tab_cons:
            st.plotly_chart(
                _make_consequence_placeholder(),
                use_container_width=True,
                key=f"cons_{comp_id}",
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
        with st.spinner("Loading seismic fragility index…"):
            seismic_objects = _get_cached_index().filter_by_hazard("seismic")

    if not seismic_objects:
        st.warning("No seismic fragility data found. Check directory structure.", icon="⚠️")
        return

    # Deduplicate file paths and pass a hashable tuple to the cached builder
    file_paths: tuple[str, ...] = tuple(
        dict.fromkeys(obj.file_path for obj in seismic_objects if obj.file_path)
    )
    tree = _build_tree(file_paths)

    # Pre-load full JSON for detail panels (also cached)
    full_json_cache: Dict[str, dict] = {}
    load_errors: List[str] = []
    for fp in file_paths:
        try:
            full_json_cache[fp] = _load_full_json(fp)
        except Exception as exc:
            load_errors.append(f"`{fp}`: {exc}")

    if load_errors:
        with st.expander("⚠️ Data load warnings", expanded=False):
            for err in load_errors:
                st.warning(err)

    total = sum(_count_components(src["groups"]) for src in tree.values())

    # ══ Root header ══════════════════════════════════════════════════════════
    st.markdown("## 🌍 Seismic")
    st.caption(f"{len(tree)} source{'s' if len(tree) != 1 else ''} · {total:,} components")
    st.divider()

    for short_name, source_data in tree.items():
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]
        groups: Dict[str, dict] = source_data["groups"]
        category: str = source_data["category"]
        full_json: dict = full_json_cache.get(fp, {})
        fname = Path(fp).name if fp else "unknown"

        badge = _CATEGORY_BADGE.get(category, f"📁 {category or 'Other'}")
        n_comp = _count_components(groups)

        # ══ Level 2: Source ═════════════════════════════════════════════════
        with st.expander(
            f"**{short_name}**  ·  {badge}  ·  `{n_comp:,}` components",
            expanded=False,
        ):
            if meta.get("Description"):
                st.caption(meta["Description"])
            st.caption(f"Version: {meta.get('Version', '—')}  ·  File: `{fname}`")
            st.divider()

            for group_name, group_data in groups.items():
                group_total = sum(
                    len(sg["components"]) for sg in group_data["subgroups"].values()
                )
                if group_total == 0:
                    continue

                # ══ Level 3: Component group ════════════════════════════════
                with st.expander(
                    f"📂  **{group_name}**  ·  `{group_total}` components",
                    expanded=False,
                ):
                    for sg_name, sg_data in group_data["subgroups"].items():
                        comps: List[str] = sorted(sg_data["components"])
                        if not comps:
                            continue

                        n_sg = len(comps)

                        # ══ Level 4: Sub-group ══════════════════════════════
                        with st.expander(
                            f"📄  {sg_name}  ·  `{n_sg}` component{'s' if n_sg != 1 else ''}",
                            expanded=False,
                        ):
                            for comp_id in comps:
                                comp_data: dict = full_json.get(comp_id, {})
                                raw_desc: str = comp_data.get(
                                    "Description",
                                    source_data["search_dict"].get(comp_id, ""),
                                )
                                preview = raw_desc[:90] + "…" if len(raw_desc) > 90 else raw_desc

                                # ══ Level 5: Component leaf ════════════════
                                # The expander label is cheap — just text.
                                # Detail content is guarded by session state
                                # so it only renders when actually opened.
                                is_open = comp_id in st.session_state[_EXPANDED_KEY]

                                with st.expander(
                                    f"🔩  **{comp_id}**  ·  {preview}",
                                    expanded=is_open,
                                ):
                                    # Toggle open state on each re-run where
                                    # the expander is visible — Streamlit has
                                    # no direct on_change for expanders, so we
                                    # use a button to explicitly load detail.
                                    if not is_open:
                                        if st.button(
                                            "Load component detail",
                                            key=f"load_{comp_id}",
                                            type="secondary",
                                        ):
                                            st.session_state[_EXPANDED_KEY].add(comp_id)
                                            st.rerun()
                                    else:
                                        # Close button to free the widget tree
                                        if st.button(
                                            "Close",
                                            key=f"close_{comp_id}",
                                            type="secondary",
                                        ):
                                            st.session_state[_EXPANDED_KEY].discard(comp_id)
                                            st.rerun()

                                        if comp_data:
                                            _render_component_detail(comp_id, comp_data)
                                        else:
                                            st.warning(
                                                f"Full data for `{comp_id}` was not found "
                                                f"in `{fname}`.",
                                                icon="⚠️",
                                            )