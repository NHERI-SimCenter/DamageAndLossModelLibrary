"""
Fuzzy search visualization module for Streamlit app.

This module provides a search interface with fuzzy matching capabilities
and filtering options for the Damage and Loss Model Library.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from rapidfuzz import fuzz, process
import json

# Import the search preparation module
from st_search.fuzzy_search_prep import FuzzyIndex, SearchObject


@dataclass
class SearchResult:
    """Container for search results with relevance scoring."""
    search_object: SearchObject
    score: float
    matched_field: str  # 'short_name', 'description', or component ID
    matched_text: str


class FuzzySearchUI:
    """
    Streamlit UI component for fuzzy searching the DLML database.
    """
    
    def __init__(self, fuzzy_index: Optional[FuzzyIndex] = None):
        """
        Initialize the search UI.
        
        Parameters
        ----------
        fuzzy_index : Optional[FuzzyIndex]
            Pre-built index, or None to build from scratch
        """
        if fuzzy_index is None:
            with st.spinner("Building search index..."):
                self.index = FuzzyIndex()
        else:
            self.index = fuzzy_index
        
        # Initialize session state for search history and selections
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'selected_components' not in st.session_state:
            st.session_state.selected_components = set()
        if 'current_search' not in st.session_state:
            st.session_state.current_search = ""
    
    def fuzzy_search(self, 
                     query: str, 
                     limit: int = 10,
                     threshold: int = 60,
                     search_fields: List[str] = None) -> List[SearchResult]:
        """
        Perform fuzzy search across multiple fields.
        
        Parameters
        ----------
        query : str
            Search query string
        limit : int
            Maximum number of results to return
        threshold : int
            Minimum similarity score (0-100)
        search_fields : List[str]
            Fields to search in ['short_names', 'descriptions', 'component_ids']
        
        Returns
        -------
        List[SearchResult]
            Sorted list of search results
        """
        if not query:
            return []
        
        if search_fields is None:
            search_fields = ['short_names', 'descriptions', 'component_ids']
        
        results = []
        seen_objects = set()  # Track unique search objects
        
        # Search in short names
        if 'short_names' in search_fields:
            short_names = [(obj.short_name, obj) for obj in self.index.search_objects]
            matches = process.extract(
                query, 
                [name for name, _ in short_names],
                scorer=fuzz.WRatio,
                limit=limit
            )
            
            for match_text, score, idx in matches:
                if score >= threshold:
                    obj = short_names[idx][1]
                    if id(obj) not in seen_objects:
                        results.append(SearchResult(
                            search_object=obj,
                            score=score,
                            matched_field='short_name',
                            matched_text=match_text
                        ))
                        seen_objects.add(id(obj))
        
        # Search in descriptions
        if 'descriptions' in search_fields:
            descriptions = [(obj.description, obj) for obj in self.index.search_objects]
            matches = process.extract(
                query,
                [desc for desc, _ in descriptions],
                scorer=fuzz.partial_ratio,
                limit=limit
            )
            
            for match_text, score, idx in matches:
                if score >= threshold:
                    obj = descriptions[idx][1]
                    if id(obj) not in seen_objects:
                        results.append(SearchResult(
                            search_object=obj,
                            score=score * 0.9,  # Slightly lower weight for description matches
                            matched_field='description',
                            matched_text=match_text[:100] + '...' if len(match_text) > 100 else match_text
                        ))
                        seen_objects.add(id(obj))
        
        # Search in component IDs
        if 'component_ids' in search_fields:
            for obj in self.index.search_objects:
                component_ids = list(obj.search_dict.keys())
                if component_ids:
                    matches = process.extract(
                        query,
                        component_ids,
                        scorer=fuzz.WRatio,
                        limit=5
                    )
                    
                    for match_text, score, _ in matches:
                        if score >= threshold and id(obj) not in seen_objects:
                            results.append(SearchResult(
                                search_object=obj,
                                score=score * 0.95,  # Component ID matches are very relevant
                                matched_field='component_id',
                                matched_text=match_text
                            ))
                            seen_objects.add(id(obj))
                            break  # Only take the best match per object
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def render_search_bar(self) -> Optional[str]:
        """
        Render the main search bar with autocomplete suggestions.
        
        Returns
        -------
        Optional[str]
            The current search query
        """
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Search input with placeholder
            search_query = st.text_input(
                "🔍 Search",
                value=st.session_state.current_search,
                placeholder="Search by component name, ID, or description...",
                key="search_input",
                help="Use fuzzy search to find components. Try typing partial names or IDs."
            )
        
        with col2:
            # Clear button
            if st.button("Clear", type="secondary", use_container_width=True):
                st.session_state.current_search = ""
                st.rerun()
        
        # Update session state
        if search_query != st.session_state.current_search:
            st.session_state.current_search = search_query
        
        return search_query if search_query else None
    
    def render_filters(self) -> Dict[str, any]:
        """
        Render filter options in an expander.
        
        Returns
        -------
        Dict[str, any]
            Dictionary of active filters
        """
        filters = {}
        
        with st.expander("🎯 Advanced Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Hazard type filter
                hazard_options = ['All', 'Seismic', 'Hurricane', 'Flood']
                hazard = st.selectbox(
                    "Hazard Type",
                    hazard_options,
                    help="Filter by hazard type"
                )
                if hazard != 'All':
                    filters['hazard'] = hazard.lower()
                
                # Search fields selection
                search_fields = st.multiselect(
                    "Search in",
                    ['short_names', 'descriptions', 'component_ids'],
                    default=['short_names', 'descriptions', 'component_ids'],
                    help="Select which fields to search"
                )
                filters['search_fields'] = search_fields
            
            with col2:
                # Similarity threshold
                threshold = st.slider(
                    "Match Threshold",
                    min_value=40,
                    max_value=100,
                    value=60,
                    step=5,
                    help="Minimum similarity score for matches (higher = stricter)"
                )
                filters['threshold'] = threshold
                
                # Results limit
                limit = st.number_input(
                    "Max Results",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Maximum number of results to display"
                )
                filters['limit'] = limit
            
            with col3:
                # Component group filter (if available)
                summary = self.index.get_summary()
                st.metric("Total Components", summary['total_components'])
                st.metric("Total Collections", summary['total_files'])
        
        return filters
    
    def render_search_results(self, results: List[SearchResult]):
        """
        Render search results in a user-friendly format.
        
        Parameters
        ----------
        results : List[SearchResult]
            List of search results to display
        """
        if not results:
            st.info("No results found. Try adjusting your search query or filters.")
            return
        
        st.subheader(f"Found {len(results)} result{'s' if len(results) != 1 else ''}")
        
        for i, result in enumerate(results):
            with st.container():
                # Create a card-like display for each result
                col1, col2, col3 = st.columns([0.7, 2.3, 1])
                
                with col1:
                    # Relevance score with color coding
                    score_color = self._get_score_color(result.score)
                    st.markdown(
                        f"<div style='text-align: center; padding: 10px; "
                        f"background-color: {score_color}; border-radius: 5px;'>"
                        f"<b>Match: {result.score:.0f}%</b></div>",
                        unsafe_allow_html=True
                    )
                    
                    # Match type badge
                    match_badge = {
                        'short_name': '📌 Name',
                        'description': '📝 Description',
                        'component_id': '🔧 Component'
                    }.get(result.matched_field, '❓ Other')
                    st.caption(match_badge)
                
                with col2:
                    # Result details
                    st.markdown(f"**{result.search_object.short_name}**")
                    
                    # Show matched text with highlighting
                    if result.matched_field == 'component_id':
                        st.caption(f"Component: {result.matched_text}")
                    
                    # Description preview
                    desc = result.search_object.description
                    if desc:
                        preview = desc[:150] + '...' if len(desc) > 150 else desc
                        st.caption(preview)
                    
                    # File path for reference
                    file_path = Path(result.search_object.file_path)
                    hazard_type = 'Unknown'
                    for hazard in ['seismic', 'hurricane', 'flood']:
                        if hazard in str(file_path).lower():
                            hazard_type = hazard.capitalize()
                            break
                    
                    st.caption(f"📁 {hazard_type} | {file_path.parent.name}")
                
                with col3:
                    # Action buttons
                    if st.button(f"View Details", key=f"view_{i}", use_container_width=True):
                        self._show_detail_modal(result.search_object)
                    
                    # Selection checkbox
                    is_selected = result.search_object.file_path in st.session_state.selected_components
                    if st.checkbox(
                        "Select",
                        value=is_selected,
                        key=f"select_{i}"
                    ):
                        st.session_state.selected_components.add(result.search_object.file_path)
                    elif not is_selected and result.search_object.file_path in st.session_state.selected_components:
                        st.session_state.selected_components.remove(result.search_object.file_path)
                
                st.divider()
    
    def render_suggestions(self, query: str, limit: int = 5) -> Optional[str]:
        """
        Render live search suggestions as the user types.
        
        Parameters
        ----------
        query : str
            Current search query
        limit : int
            Maximum number of suggestions
        
        Returns
        -------
        Optional[str]
            Selected suggestion, if any
        """
        if len(query) < 2:  # Don't show suggestions for very short queries
            return None
        
        # Get quick suggestions
        suggestions = self.fuzzy_search(query, limit=limit, threshold=50)
        
        if suggestions:
            selected = None
            st.caption("💡 Suggestions:")
            
            suggestion_cols = st.columns(min(len(suggestions), 3))
            for i, (col, suggestion) in enumerate(zip(suggestion_cols, suggestions[:3])):
                with col:
                    if st.button(
                        f"{suggestion.search_object.short_name[:30]}...",
                        key=f"suggest_{i}",
                        use_container_width=True,
                        help=f"Score: {suggestion.score:.0f}%"
                    ):
                        selected = suggestion.search_object.short_name
            
            return selected
        
        return None
    
    def render_selected_components(self):
        """Render the list of selected components."""
        if st.session_state.selected_components:
            with st.expander(f"📋 Selected Components ({len(st.session_state.selected_components)})", expanded=False):
                for path in st.session_state.selected_components:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(Path(path).parent.name)
                    with col2:
                        if st.button("Remove", key=f"remove_{path}"):
                            st.session_state.selected_components.remove(path)
                            st.rerun()
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export Selection", use_container_width=True, type="primary"):
                        self._export_selection()
                with col2:
                    if st.button("Clear All", use_container_width=True):
                        st.session_state.selected_components.clear()
                        st.rerun()
    
    def _show_detail_modal(self, search_object: SearchObject):
        """
        Display detailed information about a search object.
        
        Parameters
        ----------
        search_object : SearchObject
            The object to display details for
        """
        with st.expander(f"📊 Details: {search_object.short_name}", expanded=True):
            # General information
            st.subheader("General Information")
            for key, value in search_object.general_info_dict.items():
                if key != 'ComponentGroups':
                    st.write(f"**{key}:** {value}")
            
            # Component groups if available
            if 'ComponentGroups' in search_object.general_info_dict:
                st.subheader("Component Groups")
                groups = search_object.general_info_dict['ComponentGroups']
                if isinstance(groups, dict):
                    for group_name, members in groups.items():
                        st.write(f"**{group_name}:**")
                        if isinstance(members, list):
                            for member in members:
                                st.write(f"  • {member}")
            
            # Components
            st.subheader("Components")
            component_df = pd.DataFrame([
                {'ID': comp_id, 'Description': desc[:100] + '...' if len(desc) > 100 else desc}
                for comp_id, desc in search_object.search_dict.items()
            ])
            
            if not component_df.empty:
                st.dataframe(component_df, use_container_width=True, hide_index=True)
            
            # File information
            st.caption(f"📁 Source: {search_object.file_path}")
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on relevance score."""
        if score >= 90:
            return "#d4edda"  # Green
        elif score >= 70:
            return "#fff3cd"  # Yellow
        elif score >= 50:
            return "#f8d7da"  # Light red
        else:
            return "#f5f5f5"  # Gray
    
    def _export_selection(self):
        """Export selected components to JSON."""
        if not st.session_state.selected_components:
            st.warning("No components selected.")
            return
        
        export_data = []
        for path in st.session_state.selected_components:
            # Find the corresponding search object
            for obj in self.index.search_objects:
                if obj.file_path == path:
                    export_data.append({
                        'file_path': path,
                        'short_name': obj.short_name,
                        'description': obj.description,
                        'components': list(obj.search_dict.keys())
                    })
                    break
        
        # Create download button
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download Selection as JSON",
            data=json_str,
            file_name="selected_components.json",
            mime="application/json"
        )
    
    def render_complete_ui(self):
        """
        Render the complete search UI.
        
        This is the main entry point for using the search interface.
        """
        st.title("🔍 DLML Component Search")
        st.markdown("Search and filter components from the Damage and Loss Model Library")
        
        # Render search bar
        query = self.render_search_bar()
        
        # Render suggestions if query exists
        if query:
            suggestion = self.render_suggestions(query)
            if suggestion:
                st.session_state.current_search = suggestion
                st.rerun()
        
        # Render filters
        filters = self.render_filters()
        
        # Render selected components
        self.render_selected_components()
        
        # Perform search if query exists
        if query:
            with st.spinner("Searching..."):
                # Apply hazard filter if specified
                search_objects = self.index.search_objects
                if 'hazard' in filters:
                    search_objects = self.index.filter_by_hazard(filters['hazard'])
                
                # Create temporary index with filtered objects
                temp_index = FuzzyIndex(search_objects=search_objects)
                temp_ui = FuzzySearchUI(temp_index)
                
                # Perform fuzzy search
                results = temp_ui.fuzzy_search(
                    query,
                    limit=filters.get('limit', 10),
                    threshold=filters.get('threshold', 60),
                    search_fields=filters.get('search_fields', ['short_names', 'descriptions', 'component_ids'])
                )
                
                # Render results
                self.render_search_results(results)
        else:
            # Show summary when no search is active
            st.info("Enter a search query to begin exploring the component library.")
            
            # Display summary statistics
            summary = self.index.get_summary()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Collections", summary['total_files'])
            with col2:
                st.metric("Seismic", summary['seismic'])
            with col3:
                st.metric("Hurricane", summary['hurricane'])
            with col4:
                st.metric("Flood", summary['flood'])
            
            # Recent searches (if implemented)
            if st.session_state.search_history:
                st.subheader("Recent Searches")
                for recent_query in st.session_state.search_history[-5:]:
                    if st.button(f"🔁 {recent_query}", key=f"recent_{recent_query}"):
                        st.session_state.current_search = recent_query
                        st.rerun()


# Convenience function for easy integration
def create_search_interface(fuzzy_index: Optional[FuzzyIndex] = None) -> FuzzySearchUI:
    """
    Create and return a configured search interface.
    
    Parameters
    ----------
    fuzzy_index : Optional[FuzzyIndex]
        Pre-built index or None to build from scratch
    
    Returns
    -------
    FuzzySearchUI
        Configured search UI instance
    """
    return FuzzySearchUI(fuzzy_index)


# Example usage for standalone testing
if __name__ == "__main__":
    # This allows the module to be run standalone for testing
    search_ui = create_search_interface()
    search_ui.render_complete_ui()