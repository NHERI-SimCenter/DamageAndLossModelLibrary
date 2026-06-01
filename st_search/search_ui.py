"""
search_ui.py
------------
Streamlit UI for the semantic + structured component search.

Replaces the old rapidfuzz ``fuzzy_visuals`` panel. It drives
``st_search.semantic_index.SemanticIndex`` and follows the hybrid model agreed
for this library:

    * **Facets always prune the tree.** With no text query, the hazard / source /
      group selectors narrow the collapsible library tree in place (via the
      ``allowed_ids`` parameter the tree renderers now accept).
    * **A text query swaps in a ranked list.** When the user types a query, the
      tree is replaced by relevance-ranked results — each with a breadcrumb back
      to its tree location, a lazy details panel, and an "Add" button that feeds
      the same ``added_components`` list the sidebar and downloads read from.

Retrieval routing lives in the engine: dense + BM25 hybrid for descriptions,
exact/substring for component IDs.

Note on scores
~~~~~~~~~~~~~~
Hybrid description scores are RRF (reciprocal-rank-fusion) values, not cosine
similarities — meaningful for ordering, not as an absolute "match %". The result
list shows a *relative* relevance bar (normalised to the top hit), never a
percentage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import streamlit as st

from st_search.semantic_index import (
    SearchFilters,
    SemanticIndex,
    build_tree_corpus,
)
from st_core.component import add_component, is_component_added
from st_visuals.helpers_visual import load_full_json
from st_visuals.tree_visuals import render_seismic_tree, render_wind_tree

# Map UI hazard labels to the values stored on records / used by the trees.
_HAZARD_LABELS = ["All", "Seismic", "Hurricane"]
_HAZARD_TO_VALUE = {"Seismic": "seismic", "Hurricane": "hurricane"}

# Search modes shown to the user → engine mode strings.
_MODE_LABELS = ["Description", "ID"]
_MODE_TO_ENGINE = {"Description": "description", "ID": "id"}

_RESULT_LIMIT = 30


# ─── Cached engine ───────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Building the component search index…")
def get_index() -> SemanticIndex:
    """
    Build the semantic index once per process.

    Embeds the ~1.6k tree-visible components with fastembed (dense + BM25). The
    first cold start also downloads the embedding models; subsequent re-runs
    reuse this cached resource.
    """
    return SemanticIndex(build_tree_corpus("."))


# ─── Facet option helpers ────────────────────────────────────────────────────


def _hazard_value(label: str) -> Optional[str]:
    return _HAZARD_TO_VALUE.get(label)


def _source_options(index: SemanticIndex, hazard: Optional[str]) -> List[str]:
    """Distinct source library names, optionally scoped to a hazard."""
    names = {
        r.short_name
        for r in index.records
        if hazard is None or r.hazard == hazard
    }
    return sorted(names)


def _group_options(
    index: SemanticIndex, hazard: Optional[str], source: Optional[str]
) -> Dict[str, str]:
    """
    ``{display label -> group prefix}`` for the group selector, scoped to the
    current hazard / source selection so the choices stay relevant.
    """
    labels: Dict[str, str] = {}
    for r in index.records:
        if hazard is not None and r.hazard != hazard:
            continue
        if source is not None and r.short_name != source:
            continue
        labels[r.group_label] = r.group
    return dict(sorted(labels.items()))


# ─── Controls ────────────────────────────────────────────────────────────────


def render_search_controls(
    index: SemanticIndex,
) -> Tuple[str, str, SearchFilters, str]:
    """
    Render the query box, mode selector, and facet filters.

    Returns
    -------
    (query, engine_mode, filters, hazard_label)
    """
    col_query, col_mode = st.columns([4, 2])
    with col_query:
        query = st.text_input(
            "Search components",
            placeholder="Describe a component, e.g. “roof-to-wall connection” or a code like B.10.31",
            key="search_query",
            label_visibility="collapsed",
        )
    with col_mode:
        mode_label = st.radio(
            "Search by",
            _MODE_LABELS,
            horizontal=True,
            key="search_mode",
            label_visibility="collapsed",
        )

    with st.expander("🔧 Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            hazard_label = st.selectbox("Hazard", _HAZARD_LABELS, key="filter_hazard")
        hazard = _hazard_value(hazard_label)

        with c2:
            sources = _source_options(index, hazard)
            source_label = st.selectbox(
                "Source", ["All"] + sources, key="filter_source"
            )
        source = None if source_label == "All" else source_label

        with c3:
            group_map = _group_options(index, hazard, source)
            group_label = st.selectbox(
                "Component group", ["All"] + list(group_map), key="filter_group"
            )
        group = None if group_label == "All" else group_map.get(group_label)

    filters = SearchFilters(hazard=hazard, source=source, group=group)
    return query, _MODE_TO_ENGINE[mode_label], filters, hazard_label


# ─── Results ─────────────────────────────────────────────────────────────────


def _hazard_for_added(payload_hazard: str) -> str:
    """Engine hazard ('hurricane'/'seismic') → added-list hazard ('wind'/'seismic')."""
    return "wind" if payload_hazard == "hurricane" else "seismic"


def _breadcrumb(payload: dict) -> str:
    parts = [
        payload.get("short_name"),
        payload.get("group_label"),
        payload.get("subgroup_label"),
    ]
    # Drop empties and collapse the common case where group == subgroup label.
    crumbs: List[str] = []
    for p in parts:
        if p and (not crumbs or p != crumbs[-1]):
            crumbs.append(p)
    return "  ›  ".join(crumbs)


def render_results(
    index: SemanticIndex,
    query: str,
    mode: str,
    filters: SearchFilters,
) -> None:
    """Render the ranked result list for a non-empty query."""
    hits = index.search(query, mode=mode, filters=filters, limit=_RESULT_LIMIT)

    st.markdown(f"### 🔎 Results · `{len(hits)}`")
    if not hits:
        st.info(
            "No matches. Try different or fewer words, switch the search mode, "
            "or relax the filters.",
            icon="🔍",
        )
        return

    max_score = max((h.score for h in hits), default=1.0) or 1.0

    for rank, hit in enumerate(hits, 1):
        payload = hit.payload
        col_main, col_score, col_add = st.columns([7, 2, 2])

        with col_main:
            desc = payload.get("description", "")
            preview = desc[:160] + "…" if len(desc) > 160 else desc
            st.markdown(f"**`{hit.component_id}`** — {preview}")
            st.caption(_breadcrumb(payload))

        with col_score:
            st.caption(f"#{rank}")
            st.progress(min(hit.score / max_score, 1.0))

        with col_add:
            if is_component_added(hit.component_id):
                st.button(
                    "✅ Added",
                    key=f"sr_added_{rank}_{hit.component_id}",
                    disabled=True,
                    use_container_width=True,
                )
            elif st.button(
                "➕ Add",
                key=f"sr_add_{rank}_{hit.component_id}",
                use_container_width=True,
            ):
                fp = payload["file_path"]
                comp_data = load_full_json(fp).get(hit.component_id, {})
                add_component(
                    hit.component_id,
                    comp_data,
                    fp,
                    _hazard_for_added(payload.get("hazard", "")),
                )
                st.rerun()

        # Lazy detail panel. The expander body runs on every re-run regardless
        # of open state, so the leaf is gated behind a "Load details" button
        # (render_component_leaf_button) to avoid building 30 charts per run.
        with st.expander("Details & curves", expanded=False):
            _render_result_details(rank, hit)

        st.divider()


def _render_result_details(rank: int, hit) -> None:
    # Imported lazily to keep the module import graph shallow.
    from st_core.component import render_component_leaf_button

    payload = hit.payload
    fp = payload["file_path"]
    comp_data = load_full_json(fp).get(hit.component_id, {})
    render_component_leaf_button(
        hit.component_id,
        comp_data,
        fp,
        key_prefix=f"sr{rank}_",
        hazard=_hazard_for_added(payload.get("hazard", "")),
    )


# ─── Tree (browse) ───────────────────────────────────────────────────────────


def _has_component_filter(filters: SearchFilters) -> bool:
    """True when a facet narrows *within* a hazard (source/group), needing IDs."""
    return any(
        getattr(filters, f) is not None
        for f in ("source", "category", "group", "subgroup", "type")
    )


def render_library(
    index: SemanticIndex,
    hazard_label: str,
    filters: SearchFilters,
) -> None:
    """Render the browse tree(s), pruned to the active facets."""
    allowed_ids: Optional[set] = None
    if _has_component_filter(filters):
        allowed_ids = set(index.filter_only(filters))

    show_seismic = hazard_label in ("All", "Seismic")
    show_wind = hazard_label in ("All", "Hurricane")

    if show_seismic:
        render_seismic_tree(allowed_ids=allowed_ids)
    if show_wind:
        render_wind_tree(allowed_ids=allowed_ids)


# ─── Orchestrator ────────────────────────────────────────────────────────────


def render_search_and_library() -> None:
    """
    Top-level entry: search controls, then either ranked results (query) or the
    facet-pruned browse tree (no query).
    """
    index = get_index()
    query, mode, filters, hazard_label = render_search_controls(index)
    st.divider()

    if query.strip():
        render_results(index, query, mode, filters)
    else:
        render_library(index, hazard_label, filters)
