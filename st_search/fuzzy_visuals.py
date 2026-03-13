"""
Fuzzy search visualization module for Streamlit app.
Provides search bar UI with dynamic suggestions and filtering options.
"""

import streamlit as st
from rapidfuzz import fuzz, process
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
from st_search.component_search import FuzzyIndex, SearchObject
from st_keyup import st_keyup

@st.cache_resource
def _get_cached_index() -> FuzzyIndex:
    """
    Cache the FuzzyIndex to avoid repeated JSON parsing.
    
    Returns
    -------
    FuzzyIndex
        Cached index of all fragility.json files
    """
    return FuzzyIndex()


@dataclass
class SearchResult:
    """Container for search results with relevance scoring."""
    component_id: str
    description: str
    score: float
    source_object: SearchObject
    hazard_type: str = field(init=False)
    
    def __post_init__(self):
        """Extract hazard type from file path."""
        if 'seismic' in self.source_object.file_path:
            self.hazard_type = 'seismic'
        elif 'hurricane' in self.source_object.file_path:
            self.hazard_type = 'hurricane'
        elif 'flood' in self.source_object.file_path:
            self.hazard_type = 'flood'
        else:
            self.hazard_type = 'unknown'


class FuzzySearchUI:
    """Main class for handling fuzzy search UI in Streamlit."""
    
    def __init__(self):
        """Initialize the search UI with cached index."""
        self.index = _get_cached_index()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for tracking selections."""
        if 'selected_component_ids' not in st.session_state:
            st.session_state.selected_component_ids = []
        if 'selected_component_metadata' not in st.session_state:
            st.session_state.selected_component_metadata = {}
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_search_results' not in st.session_state:
            st.session_state.current_search_results = []
        if 'show_suggestions' not in st.session_state:
            st.session_state.show_suggestions = False
    
    def _store_component_metadata(self, comp_id: str, search_result: SearchResult):
        """
        Store full metadata for a selected component in session state.

        Extracts the description, comments, and limit-state info from the
        source SearchObject's underlying JSON data.

        Parameters
        ----------
        comp_id : str
            The component ID being added.
        search_result : SearchResult
            The SearchResult containing the source object with full JSON data.
        """
        source_obj = search_result.source_object

        # The source object's combined_dict has _GeneralInformation removed
        # into general_info_dict; the remaining keys are component entries.
        # Each component entry in the original JSON may be a dict with
        # Description, Comments, LimitStates, etc., or just a description string.
        raw_component_data = source_obj.combined_dict.get(comp_id, {})

        if isinstance(raw_component_data, dict):
            description = raw_component_data.get('Description', search_result.description)
            comments = raw_component_data.get('Comments', '')
            limit_states = raw_component_data.get('LimitStates', {})
            block_size = raw_component_data.get('SuggestedComponentBlockSize', '')
        else:
            # Fallback: the search_dict value is just the description string
            description = search_result.description
            comments = ''
            limit_states = {}
            block_size = ''

        st.session_state.selected_component_metadata[comp_id] = {
            'description': description,
            'comments': comments,
            'limit_states': limit_states,
            'block_size': block_size,
            'hazard_type': search_result.hazard_type,
            'source_file': source_obj.file_path,
            'short_name': source_obj.short_name,
        }

    def _fuzzy_search_components(
        self, 
        query: str, 
        limit: int = 10,
        score_cutoff: int = 60,
        filtered_objects: Optional[List[SearchObject]] = None,
        search_mode: str = "Description",
    ) -> List[SearchResult]:
        """
        Perform fuzzy search on component descriptions, IDs, or titles.
        
        Parameters
        ----------
        query : str
            Search query string
        limit : int
            Maximum number of results to return
        score_cutoff : int
            Minimum score threshold (0-100)
        filtered_objects : Optional[List[SearchObject]]
            Pre-filtered list of SearchObjects to search within
        search_mode : str
            One of ``"ID"``, ``"Title"``, or ``"Description"`` (default).
            Controls which field the fuzzy matcher runs against.
        
        Returns
        -------
        List[SearchResult]
            Sorted list of search results
        """
        if not query or not query.strip():
            return []
        
        search_pool = filtered_objects if filtered_objects else self.index.search_objects
        results = []
        
        for search_obj in search_pool:
            # Create a list of tuples (component_id, description) for this object
            component_items: list[tuple[str, str]] = list(search_obj.search_dict.items())

            # ── Build the list of choices based on search_mode ──
            if search_mode == "ID":
                # Match against the component ID strings
                choices = [comp_id for comp_id, _ in component_items]
            elif search_mode == "Title":
                # Match against the parent SearchObject's short name.
                # Every component under the same JSON file shares the same
                # title, so we repeat it per-item to keep index alignment.
                choices = [search_obj.short_name] * len(component_items)
            else:
                # Default: match against the component description text
                choices = [desc for _, desc in component_items]

            matches = process.extract(
                query = query,
                choices = choices,
                scorer = fuzz.WRatio,
                limit = limit,
                processor=str.casefold
            )

            # Create SearchResult objects for matches above threshold
            for match_text, score, idx in matches:
                if score >= score_cutoff:
                    component_id = component_items[idx][0]
                    description = component_items[idx][1]
                    results.append(
                        SearchResult(
                            component_id=component_id,
                            description=description,
                            score=score,
                            source_object=search_obj
                        )
                    )
        
        # Sort by score (highest first) and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _render_filters(self) -> Tuple[Optional[str], Optional[str], Set[str], int]:
        """
        Render filter UI components.
        
        Returns
        -------
        Tuple[Optional[str], Optional[str], Set[str], int]
            Selected hazard type, component group, additional filters, and score threshold
        """
        with st.expander("🔧 Search Filters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # ── Assessment methodology filter (UI only — filtering logic TBD) ──
                methodology_options = ['All', 'FEMA P-58', 'Hazus v5.1', 'Other']
                st.selectbox(
                    "Assessment Methodology",
                    methodology_options,
                    index=0,
                    key="methodology_filter",
                    help=(
                        "Filter components by their source assessment methodology. "
                        "Filtering logic is not yet implemented — selecting a value "
                        "here currently has no effect on search results."
                    ),
                )

                # Hazard type filter
                hazard_options = ['All', 'Seismic', 'Hurricane', 'Flood']
                selected_hazard = st.selectbox(
                    "Hazard Type",
                    hazard_options,
                    index=0,
                    key="hazard_filter"
                )
                
                # Component group filter (if available)
                all_groups = set()
                for obj in self.index.search_objects:
                    groups = obj.general_info_dict.get('ComponentGroups', {})
                    if isinstance(groups, dict):
                        all_groups.update(groups.keys())
                
                group_options = ['All'] + sorted(list(all_groups))
                selected_group = st.selectbox(
                    "Component Group",
                    group_options,
                    index=0,
                    key="group_filter"
                ) if all_groups else None
            
            with col2:
                # Additional filter options
                filter_options = st.multiselect(
                    "Additional Filters",
                    ['Include incomplete data', 'Show anchored only', 'Show unanchored only'],
                    key="additional_filters"
                )
                
                # Score threshold slider
                score_threshold = st.slider(
                    "Minimum Match Score",
                    min_value=0,
                    max_value=100,
                    value=60,
                    step=5,
                    key="score_threshold",
                    help="Higher values return more precise matches"
                )
        
        hazard = None if selected_hazard == 'All' else selected_hazard.lower()
        group = None if selected_group == 'All' else selected_group
        
        return hazard, group, set(filter_options), score_threshold
    
    def _apply_filters(
        self, 
        hazard: Optional[str] = None,
        component_group: Optional[str] = None,
        additional_filters: Optional[Set[str]] = None
    ) -> List[SearchObject]:
        """
        Apply filters to the search pool.
        
        Parameters
        ----------
        hazard : Optional[str]
            Hazard type to filter by
        component_group : Optional[str]
            Component group to filter by
        additional_filters : Set[str]
            Additional filter criteria
        
        Returns
        -------
        List[SearchObject]
            Filtered list of SearchObjects
        """
        filtered = self.index.search_objects.copy()
        
        # Apply hazard filter
        if hazard:
            filtered = self.index.filter_by_hazard(hazard)
        
        # Apply component group filter
        if component_group:
            filtered = [obj for obj in filtered 
                       if component_group in str(obj.general_info_dict.get('ComponentGroups', {}))]
        
        # Apply additional filters
        if additional_filters:
            if 'Show anchored only' in additional_filters:
                filtered = [obj for obj in filtered 
                           if any('.A' in comp_id or '.A.' in comp_id 
                                 for comp_id in obj.search_dict.keys())]
            
            if 'Show unanchored only' in additional_filters:
                filtered = [obj for obj in filtered 
                           if any('.U' in comp_id or '.U.' in comp_id 
                                 for comp_id in obj.search_dict.keys())]
        
        return filtered
    
    def _render_search_suggestions(self, query: str, results: List[SearchResult]):
        """
        Render dynamic search suggestions dropdown.
        
        Parameters
        ----------
        query : str
            Current search query
        results : List[SearchResult]
            Search results to display as suggestions
        """
        if not results or not query:
            return
        
        # Create a container for suggestions
        suggestion_container = st.container()
        
        with suggestion_container:
            st.markdown("### 💡 Suggestions")
            
            # Display top 5 suggestions
            for i, result in enumerate(results[:5]):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Make suggestion clickable
                    if st.button(
                        f"{result.component_id}: {result.description[:50]}...",
                        key=f"suggestion_{i}",
                        use_container_width=True
                    ):
                        # Add to selected components and store metadata
                        if result.component_id not in st.session_state.selected_component_ids:
                            st.session_state.selected_component_ids.append(result.component_id)
                            self._store_component_metadata(result.component_id, result)
                            st.rerun()
                
                with col2:
                    # Show match score
                    st.metric("Match", f"{result.score:.0f}%", label_visibility="collapsed")
                
                with col3:
                    # Show hazard type with emoji
                    hazard_emoji = {
                        'seismic': '🏢',
                        'hurricane': '🌀',
                        'flood': '💧',
                        'unknown': '❓'
                    }
                    st.write(f"{hazard_emoji.get(result.hazard_type, '❓')} {result.hazard_type.title()}")
    
    def _render_selected_components(self):
        """Render the list of selected components with remove options and full descriptions."""
        if st.session_state.selected_component_ids:
            st.markdown("### 📌 Selected Components")
            
            # Add remove buttons for each component, with full description
            for idx, comp_id in enumerate(st.session_state.selected_component_ids):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{idx + 1}.** {comp_id}")

                    # ── Display full component description ──────────────────
                    meta = st.session_state.selected_component_metadata.get(comp_id)
                    if meta:
                        desc = meta.get('description', '')
                        comments = meta.get('comments', '')
                        block_size = meta.get('block_size', '')

                        if desc:
                            st.caption(f"**Description:** {desc}")
                        if comments:
                            st.caption(f"**Comments:** {comments}")
                        if block_size:
                            st.caption(f"**Block Size:** {block_size}")
                    else:
                        st.caption(
                            "_No description available — component was "
                            "added before metadata tracking was enabled._"
                        )
                    # ────────────────────────────────────────────────────────
                
                with col2:
                    if st.button("Remove", key=f"remove_{comp_id}"):
                        st.session_state.selected_component_ids.remove(comp_id)
                        st.session_state.selected_component_metadata.pop(comp_id, None)
                        st.rerun()
            
            # Clear all button
            if st.button("🗑️ Clear All Selections", type="secondary"):
                st.session_state.selected_component_ids.clear()
                st.session_state.selected_component_metadata.clear()
                st.rerun()
        else:
            st.info("No components selected yet. Use the search bar to find and select components.")
    
    def render_search_interface(self):
        """
        Main method to render the complete search interface.
        
        Returns
        -------
        List[str]
            List of selected component IDs
        """
        st.markdown("## 🔍 Component Search")
        
        # Render filters
        hazard_filter, group_filter, additional_filters, score_threshold = self._render_filters()
        
        # Apply filters to get filtered search pool
        filtered_objects = self._apply_filters(hazard_filter, group_filter, additional_filters)
        
        # Search bar
        search_mode = st.radio(
            "Search by",
            options=["Description", "ID", "Title"],
            index=0,
            horizontal=True,
            key="search_mode",
        )

        placeholder_map = {
            "Description": "Enter component description or keywords...",
            "ID": "Enter component ID (e.g. B.10.31.001)...",
            "Title": "Enter model library title or short name...",
        }

        col1, col2 = st.columns([5, 1])
        
        with col1:
            search_query = st_keyup(
                label="Search for components",
                placeholder=placeholder_map.get(search_mode, "Search..."),
                key="search_input",
                label_visibility="collapsed",
            )

        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True, key='search_button')
        
        # Perform search when query changes or button is clicked

        if search_query and (search_button or search_query != st.session_state.get('last_query', '')):
            st.session_state.last_query = search_query
            
            # Perform fuzzy search
            results = self._fuzzy_search_components(
                search_query,
                limit=50,
                score_cutoff=score_threshold,
                filtered_objects=filtered_objects,
                search_mode=search_mode,
            )
            
            st.session_state.current_search_results = results
            
            # Add to search history
            if search_query not in st.session_state.search_history:
                st.session_state.search_history.insert(0, search_query)
                st.session_state.search_history = st.session_state.search_history[:10]
    
        # Display search results
        if st.session_state.current_search_results:
            st.markdown("### 🎯 Search Results")
            
            results_df = pd.DataFrame([
                {
                    'Component ID': r.component_id,
                    'Description': r.description[:100] + '...' if len(r.description) > 100 else r.description,
                    'Match Score': f"{r.score:.1f}%",
                    'Hazard Type': r.hazard_type.title()
                }
                for r in st.session_state.current_search_results[:50]
            ])
            
            # Make results selectable
            selected_indices = st.multiselect(
                "Select components to add:",
                options=list(range(len(results_df))),
                format_func=lambda x: f"{results_df.iloc[x]['Component ID']}: {results_df.iloc[x]['Description'][:50]}...",
                key="result_selector"
            )
            
            if selected_indices:
                if st.button("➕ Add Selected Components", type="primary"):
                    for idx in selected_indices:
                        result = st.session_state.current_search_results[idx]
                        comp_id = result.component_id
                        if comp_id not in st.session_state.selected_component_ids:
                            st.session_state.selected_component_ids.append(comp_id)
                            self._store_component_metadata(comp_id, result)
                    st.rerun()
            
            # Display results table
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Display selected components
        st.divider()
        self._render_selected_components()
        
        # Search history
        if st.session_state.search_history:
            with st.expander("📜 Recent Searches"):
                for query in st.session_state.search_history:
                    if st.button(query, key=f"history_{query}", use_container_width=True):
                        st.session_state.search_input = query
                        st.rerun()
        
        return st.session_state.selected_component_ids


# Convenience function for easy integration
def render_fuzzy_search() -> List[str]:
    """
    Render the fuzzy search interface and return selected component IDs.
    
    Returns
    -------
    List[str]
        List of selected component IDs
    """
    search_ui = FuzzySearchUI()
    return search_ui.render_search_interface()