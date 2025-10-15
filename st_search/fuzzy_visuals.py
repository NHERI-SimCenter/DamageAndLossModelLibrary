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
    """[]
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
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'current_search_results' not in st.session_state:
            st.session_state.current_search_results = []
        if 'show_suggestions' not in st.session_state:
            st.session_state.show_suggestions = False
    
    def _fuzzy_search_components(
        self, 
        query: str, 
        limit: int = 10,
        score_cutoff: int = 60,
        filtered_objects: Optional[List[SearchObject]] = None
    ) -> List[SearchResult]:
        """
        Perform fuzzy search on component descriptions.
        
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

            # Search through descriptions using rapidfuzz
            descriptions = [temp_descsciption for temp_component_id, temp_descsciption in component_items]
            matches = process.extract(
                query = query,
                choices = descriptions,
                scorer = fuzz.WRatio,
                limit = limit, #min(limit, len(descriptions)),
                processor=str.casefold
            )

            # Create SearchResult objects for matches above threshold
            for match_desc, score, idx in matches:
                if score >= score_cutoff:
                    component_id = component_items[idx][0]
                    results.append(
                        SearchResult(
                            component_id=component_id,
                            description=match_desc,
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
                        # Add to selected components
                        if result.component_id not in st.session_state.selected_component_ids:
                            st.session_state.selected_component_ids.append(result.component_id)
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
        """Render the list of selected components with remove options."""
        if st.session_state.selected_component_ids:
            st.markdown("### 📌 Selected Components")
            
            # Create a DataFrame for better display
            selected_df = pd.DataFrame(
                st.session_state.selected_component_ids,
                columns=['Component ID']
            )
            
            # Add remove buttons for each component
            for idx, comp_id in enumerate(st.session_state.selected_component_ids):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{idx + 1}.** {comp_id}")
                
                with col2:
                    if st.button("Remove", key=f"remove_{comp_id}"):
                        st.session_state.selected_component_ids.remove(comp_id)
                        st.rerun()
            
            # Clear all button
            if st.button("🗑️ Clear All Selections", type="secondary"):
                st.session_state.selected_component_ids.clear()
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
        col1, col2 = st.columns([5, 1])
        
        with col1:
            search_query = st_keyup(
                label="Search for components",
                placeholder="Enter component description, ID, or keywords...",
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
                filtered_objects=filtered_objects
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
                        comp_id = st.session_state.current_search_results[idx].component_id
                        if comp_id not in st.session_state.selected_component_ids:
                            st.session_state.selected_component_ids.append(comp_id)
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