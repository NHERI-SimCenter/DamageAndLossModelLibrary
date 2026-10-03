"""
search_ui.py
------------
Streamlit UI for the semantic + structured component search.

Replaces the old rapidfuzz ``fuzzy_visuals`` panel. It drives
``dlml.web.st_search.semantic_index.SemanticIndex`` and follows the hybrid
model agreed for this library:

    * **Facets narrow the tree.** With no text query, the Hazard selector
      chooses which hazard sections of the collapsible library tree render, and
      the Asset type and Source selectors narrow them to the matching data files
      (via the ``allowed_files`` parameter of the tree renderers).
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
list shows a *relative* relevance bar (normalized to the top hit), never a
percentage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import streamlit as st

from dlml.web.st_search.semantic_index import (
    SearchFilters,
    SemanticIndex,
    build_tree_corpus,
)
from dlml.web.st_core.component import add_component, is_component_added
from dlml.web.st_visuals.helpers_visual import load_full_json
from dlml.web.st_visuals.tree_visuals import (
    asset_type_label,
    render_consequence_tree,
    render_seismic_tree,
    render_wind_tree,
)

# Map UI hazard labels to the values stored on records / used by the trees.
_HAZARD_LABELS = ["All", "Seismic", "Hurricane"]
_HAZARD_TO_VALUE = {"Seismic": "seismic", "Hurricane": "hurricane"}

# Search modes shown to the user → engine mode strings.
_MODE_LABELS = ["Description", "ID"]
_MODE_TO_ENGINE = {"Description": "description", "ID": "id"}

# Collection selector: scopes both search and browse. "All" spans both
# collections.
_DATASET_LABELS = ["Fragility", "Consequence", "All"]
_DATASET_TO_VALUE = {"Fragility": "fragility", "Consequence": "consequence", "All": None}
_DATASET_BADGE = {"fragility": "🔧", "consequence": "🧾"}

_RESULT_LIMIT = 30


# ─── Cached engine ───────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Building the component search index…")
def get_index() -> SemanticIndex:
    """
    Build the semantic index once per process.

    Indexes the tree-visible records, both fragility (damage) and consequence
    (repair) models, and embeds all of them except the Hazus hurricane
    portfolio records with fastembed (dense + BM25); those are found by ID and
    facet search only. The first cold start also downloads the embedding
    models; subsequent re-runs reuse this cached resource.
    """
    return SemanticIndex(build_tree_corpus())


# ─── Facet option helpers ────────────────────────────────────────────────────


def _hazard_value(label: str) -> Optional[str]:
    return _HAZARD_TO_VALUE.get(label)


def _asset_type_options(
    index: SemanticIndex, hazard: Optional[str], dataset: Optional[str]
) -> Dict[str, str]:
    """
    Map display labels to asset types for the Asset type selector.

    Covers the asset types present in the index for the hazard and collection
    selection, sorted by label.
    """
    scope = SearchFilters(hazard=hazard, dataset=dataset)
    asset_types = {
        row["asset_type"]
        for row in index.file_facets
        if row["asset_type"] and scope.matches(row)
    }
    return dict(sorted((asset_type_label(a), a) for a in asset_types))


def _source_options(
    index: SemanticIndex,
    hazard: Optional[str],
    asset_type: Optional[str],
    dataset: Optional[str],
) -> List[str]:
    """
    Distinct source titles for the Source selector, sorted case-insensitively.

    Scoped to the hazard, asset type and collection selection.
    """
    scope = SearchFilters(hazard=hazard, asset_type=asset_type, dataset=dataset)
    names = {row["short_name"] for row in index.file_facets if scope.matches(row)}
    return sorted(names, key=str.casefold)


def _reset_if_missing(key: str, options: List[str]) -> None:
    """
    Reset a selectbox's stored value to "All" when the value is no longer among the options.

    Called before the widget is created in each script run, so the widget
    starts from a valid value when an upstream selection removed the stored
    one.
    """
    if st.session_state.get(key, "All") not in options:
        st.session_state[key] = "All"


def _clear_search_query() -> None:
    """
    Empty the search box (used as a button ``on_click`` callback).

    Callbacks run before widgets are instantiated on the next rerun, so
    resetting the ``search_query`` state here is allowed — and with an empty
    query the page falls back to the browse tree.
    """
    st.session_state["search_query"] = ""


# ─── Controls ────────────────────────────────────────────────────────────────


def render_search_controls(
    index: SemanticIndex,
    dataset: Optional[str],
) -> Tuple[str, str, SearchFilters, str]:
    """
    Render the query box, mode selector, and facet filters.

    ``dataset`` (from the Collection selector) scopes the asset type and source
    options and is included in the returned filters so search and browse stay
    in sync. Each filter's options follow the selections before it (Hazard,
    then Asset type, then Source); a stored value that is no longer offered
    falls back to "All".

    Returns
    -------
    (query, engine_mode, filters, hazard_label)
    """
    col_query, col_clear, col_mode = st.columns([5, 1, 2])
    with col_query:
        query = st.text_input(
            "Search components",
            placeholder="Describe a model, e.g. “roof-to-wall connection” or a code like B.10.31",
            key="search_query",
            label_visibility="collapsed",
        )
    with col_clear:
        st.button(
            "✕ Clear",
            key="clear_search",
            on_click=_clear_search_query,
            disabled=not query.strip(),
            width="stretch",
            help="Clear the search and return to the browse tree",
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
            asset_type_map = _asset_type_options(index, hazard, dataset)
            asset_type_labels = ["All", *asset_type_map]
            _reset_if_missing("filter_asset_type", asset_type_labels)
            asset_type_choice = st.selectbox(
                "Asset type", asset_type_labels, key="filter_asset_type"
            )
        asset_type = asset_type_map.get(asset_type_choice)

        with c3:
            source_labels = [
                "All",
                *_source_options(index, hazard, asset_type, dataset),
            ]
            _reset_if_missing("filter_source", source_labels)
            source_label = st.selectbox("Source", source_labels, key="filter_source")
        source = None if source_label == "All" else source_label

    filters = SearchFilters(
        hazard=hazard, asset_type=asset_type, source=source, dataset=dataset
    )
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
            badge = _DATASET_BADGE.get(payload.get("dataset", ""), "")
            st.markdown(f"{badge} **`{hit.component_id}`** — {preview}")
            st.caption(_breadcrumb(payload))

        with col_score:
            st.caption(f"#{rank}")
            st.progress(min(hit.score / max_score, 1.0))

        with col_add:
            if is_component_added(hit.component_id, payload.get("file_path")):
                st.button(
                    "✅ Added",
                    key=f"sr_added_{rank}_{hit.component_id}",
                    disabled=True,
                    width="stretch",
                )
            elif st.button(
                "➕ Add",
                key=f"sr_add_{rank}_{hit.component_id}",
                width="stretch",
            ):
                fp = payload["file_path"]
                comp_data = load_full_json(fp).get(hit.component_id, {})
                add_component(
                    hit.component_id,
                    comp_data,
                    fp,
                    _hazard_for_added(payload.get("hazard", "")),
                    kind=payload.get("dataset", "fragility"),
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
    from dlml.web.st_core.component import render_component_leaf_button, render_consequence_leaf

    payload = hit.payload
    fp = payload["file_path"]
    cid = hit.component_id

    if payload.get("dataset") == "consequence":
        # Same lazy "Load details" guard render_component_leaf_button uses, so
        # consequence charts aren't built for every result on every re-run.
        load_key = f"sr{rank}_loaded_{cid}"
        if load_key not in st.session_state:
            if st.button("Load details", key=f"sr{rank}_btn_{cid}", type="secondary"):
                st.session_state[load_key] = True
                st.rerun()
        else:
            comp_data = load_full_json(fp).get(cid, {})
            if comp_data:
                render_consequence_leaf(cid, comp_data, fp, key_prefix=f"sr{rank}_")
        return

    comp_data = load_full_json(fp).get(cid, {})
    render_component_leaf_button(
        cid,
        comp_data,
        fp,
        key_prefix=f"sr{rank}_",
        hazard=_hazard_for_added(payload.get("hazard", "")),
    )


# ─── Tree (browse) ───────────────────────────────────────────────────────────


def _has_component_filter(filters: SearchFilters) -> bool:
    """
    True when a facet narrows *within* a data file and so needs component IDs.

    These facets are category, group, subgroup and type.
    """
    return any(
        getattr(filters, f) is not None
        for f in ("category", "group", "subgroup", "type")
    )


def render_library(
    index: SemanticIndex,
    hazard_label: str,
    filters: SearchFilters,
    dataset: Optional[str],
) -> None:
    """
    Render the browse tree(s), narrowed to the active facets.

    The Collection selector chooses which trees show (fragility, consequence,
    or both when "All") and the Hazard selector which hazard sections. Asset
    type and Source select whole data files: when either is set,
    ``allowed_files`` holds the matching files and each tree renders only those
    sources. A chosen Source also opens its expander, without opening its
    groups or rendering its models. ``allowed_ids`` is computed only for facets
    that narrow within a file.
    """
    allowed_ids: Optional[set] = None
    if _has_component_filter(filters):
        allowed_ids = set(index.filter_only(filters))
    allowed_files: Optional[set] = None
    if filters.asset_type is not None or filters.source is not None:
        allowed_files = index.allowed_files(filters)
    _render_trees(
        dataset,
        hazard_label,
        allowed_ids,
        allowed_files=allowed_files,
        expand_sources=filters.source is not None,
    )


def _render_trees(
    dataset: Optional[str],
    hazard_label: str,
    allowed_ids: Optional[set],
    *,
    allowed_files: Optional[set] = None,
    expand_sources: bool = False,
) -> None:
    """
    Render the browse tree(s) for a collection + hazard selection.

    ``allowed_files`` limits every tree to the sources whose data file is in
    the set; a tree with none of them renders nothing. ``allowed_ids`` prunes
    every tree to a shared set of component IDs and opens every level.
    ``expand_sources`` opens the source expanders.
    Pass ``None``, ``None`` and ``False`` to render the full library unpruned
    and collapsed. This is the one place the trees are laid out, shared by the
    normal facet path and the search-unavailable fallback.
    """
    show_fragility = dataset in ("fragility", None)
    show_consequence = dataset in ("consequence", None)
    show_seismic = hazard_label in ("All", "Seismic")
    show_hurricane = hazard_label in ("All", "Hurricane")

    tree_kwargs = {
        "allowed_ids": allowed_ids,
        "allowed_files": allowed_files,
        "expand_sources": expand_sources,
    }

    if show_fragility:
        if show_seismic:
            render_seismic_tree(**tree_kwargs)
        if show_hurricane:
            render_wind_tree(**tree_kwargs)

    if show_consequence:
        if show_seismic:
            render_consequence_tree("seismic", **tree_kwargs)
        if show_hurricane:
            render_consequence_tree("hurricane", **tree_kwargs)


# ─── Orchestrator ────────────────────────────────────────────────────────────


def render_search_and_library() -> None:
    """
    Top-level entry.

    A **Collection** selector (Fragility / Consequence / All) scopes both search
    and browse. Both collections are indexed, so a query searches whichever the
    selector allows; with no query, the matching browse tree(s) render. "All"
    spans both, which helps to find a model without knowing which collection it
    is in.

    If the search index cannot be built (the optional search dependencies are
    missing, or the first-run model download fails offline), the page degrades
    gracefully: a notice explains why and the full browse tree still renders.
    """
    dataset_label = st.radio(
        "Collection",
        _DATASET_LABELS,
        horizontal=True,
        key="dataset_filter",
        help=(
            "The model collection to browse and search. Fragility = damage models. "
            "Consequence = repair cost/time models (some keyed by occupancy "
            "class, with no fragility component). 'All' shows both."
        ),
    )
    dataset = _DATASET_TO_VALUE[dataset_label]

    try:
        index = get_index()
    except Exception as exc:  # noqa: BLE001 -- degrade gracefully, browsing works
        _render_search_unavailable(exc)
        st.divider()
        _render_trees(dataset, "All", None, allowed_files=None, expand_sources=False)
        return

    query, mode, filters, hazard_label = render_search_controls(index, dataset)
    st.divider()

    if query.strip():
        render_results(index, query, mode, filters)
    else:
        render_library(index, hazard_label, filters, dataset)


def _render_search_unavailable(exc: Exception) -> None:
    """
    Explain why component search is unavailable, without a traceback.

    An ``ImportError`` means the optional search dependencies are not installed;
    anything else (typically a network error while fetching the embedding models
    on first run) is reported as a transient failure. Browsing still works in
    both cases.
    """
    if isinstance(exc, ImportError):
        st.info(
            "🔍 **Component search is unavailable** because its optional "
            "dependencies are not installed. Install them with "
            "`pip install 'simcenter-dlml[explorer]'` (bundles qdrant-client + fastembed). "
            "You can still browse the full library below."
        )
    else:
        st.warning(
            "🔍 **Component search is temporarily unavailable** — the embedding "
            "models could not be loaded. The first run downloads them and needs "
            "internet access; check your connection and reload. You can still "
            "browse the full library below."
        )
