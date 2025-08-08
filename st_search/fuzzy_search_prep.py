import json
import os
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional

def extract_search_metadata(json_path: str) -> dict:
    general_info: pd.Series
    general_dict: dict
    general_dict_wrapper: str = '_GeneralInformation'
    processed_dict: dict 

    processign_df = pd.read_json(json_path)

    #Popped _GeneralInfo will be re-added to final dict for fuzzy search filtering
    general_info: pd.Series = processign_df.pop('_GeneralInformation')
    general_info = general_info.dropna()
    general_dict = general_info.to_dict()
    general_dict = {general_dict_wrapper: general_dict}

    #json description will be target of fuzzy search
    processign_df = processign_df.loc['Description']
    processed_dict = processign_df.to_dict()

    combined_dict = general_dict | processed_dict

    return combined_dict

@dataclass
class SearchObject:
    combined_dict: dict                 # required on init
    file_path: str = field(default='')  # store the source file path

    short_name: str = field(init=False)
    description: str = field(init=False)
    search_dict: dict = field(init=False)
    general_info_dict: dict = field(init=False)

    def __post_init__(self):
        temp_dict = self.combined_dict.copy()
        self.general_info_dict = temp_dict.pop('_GeneralInformation')
        
        self.short_name = self.general_info_dict.get('ShortName', 'Unknown')
        self.description = self.general_info_dict.get('Description', '')

        self.search_dict = temp_dict

    

def parse_all_fragility_json(base_path: str = '.') -> List[SearchObject]:
    """
    Parse all fragility.json files in the directory structure.
    
    Parameters
    ----------
    base_path : str
        The base directory path to search from (default is current directory)
    
    Returns
    -------
    List[SearchObject]
        A list of SearchObject instances, one for each fragility.json file found
    """
    search_objects = []
    
    # Define the hazard directories to search
    hazard_dirs = ['seismic', 'hurricane', 'flood']
    
    for hazard in hazard_dirs:
        hazard_path = Path(base_path) / hazard
        
        # Skip if hazard directory doesn't exist
        if not hazard_path.exists():
            continue
            
        # Find all fragility.json files recursively in this hazard directory
        fragility_files = hazard_path.rglob('fragility.json')
        
        for json_file in fragility_files:
            try:
                # Extract metadata from the JSON file
                metadata_dict = extract_search_metadata(str(json_file))
                
                # Create a SearchObject with the file path for reference
                search_obj = SearchObject(
                    combined_dict=metadata_dict,
                    file_path=str(json_file)
                )
                
                search_objects.append(search_obj)
                
            except Exception as e:
                # Log or handle errors for individual files
                # Continue processing other files even if one fails
                print(f"Error processing {json_file}: {e}")
                continue
    
    return search_objects


@dataclass
class FuzzyIndex:
    """
    Index for storing and searching SearchObjects from fragility.json files.
    """
    search_objects: List[SearchObject] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize the index by parsing all fragility.json files if not provided."""
        if not self.search_objects:
            self.search_objects = parse_all_fragility_json()
    
    def add_search_object(self, search_obj: SearchObject):
        """Add a single SearchObject to the index."""
        self.search_objects.append(search_obj)
    
    def get_all_short_names(self) -> List[str]:
        """Get all short names from the index."""
        return [obj.short_name for obj in self.search_objects]
    
    def get_all_descriptions(self) -> List[str]:
        """Get all descriptions from the index."""
        return [obj.description for obj in self.search_objects]
    
    def filter_by_hazard(self, hazard: str) -> List[SearchObject]:
        """
        Filter search objects by hazard type based on file path.
        
        Parameters
        ----------
        hazard : str
            The hazard type to filter by ('seismic', 'hurricane', or 'flood')
        
        Returns
        -------
        List[SearchObject]
            Filtered list of SearchObjects
        """
        filtered = []
        for obj in self.search_objects:
            if f'/{hazard}/' in obj.file_path or obj.file_path.startswith(f'{hazard}/'):
                filtered.append(obj)
        return filtered
    
    def filter_by_component_group(self, component_group: str) -> List[SearchObject]:
        """
        Filter search objects by component group if available in metadata.
        
        Parameters
        ----------
        component_group : str
            The component group to filter by
        
        Returns
        -------
        List[SearchObject]
            Filtered list of SearchObjects
        """
        filtered = []
        for obj in self.search_objects:
            component_groups = obj.general_info_dict.get('ComponentGroups', {})
            # Check if the component group exists in the ComponentGroups dict
            if component_group in str(component_groups):
                filtered.append(obj)
        return filtered
    
    def search_by_keyword(self, keyword: str, search_descriptions: bool = True, 
                         search_names: bool = True) -> List[SearchObject]:
        """
        Simple keyword search in short names and/or descriptions.
        
        Parameters
        ----------
        keyword : str
            The keyword to search for (case-insensitive)
        search_descriptions : bool
            Whether to search in descriptions (default True)
        search_names : bool
            Whether to search in short names (default True)
        
        Returns
        -------
        List[SearchObject]
            List of SearchObjects matching the keyword
        """
        keyword_lower = keyword.lower()
        results = []
        
        for obj in self.search_objects:
            match = False
            
            if search_names and keyword_lower in obj.short_name.lower():
                match = True
            
            if search_descriptions and keyword_lower in obj.description.lower():
                match = True
            
            # Also search in the component IDs (keys of search_dict)
            for component_id in obj.search_dict.keys():
                if keyword_lower in component_id.lower():
                    match = True
                    break
            
            if match:
                results.append(obj)
        
        return results
    
    def get_by_component_id(self, component_id: str) -> Optional[SearchObject]:
        """
        Get a SearchObject by exact component ID match.
        
        Parameters
        ----------
        component_id : str
            The component ID to search for
        
        Returns
        -------
        Optional[SearchObject]
            The SearchObject containing the component ID, or None if not found
        """
        for obj in self.search_objects:
            if component_id in obj.search_dict:
                return obj
        return None
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get a summary of the index contents.
        
        Returns
        -------
        Dict[str, int]
            Summary statistics of the index
        """
        summary = {
            'total_files': len(self.search_objects),
            'seismic': len(self.filter_by_hazard('seismic')),
            'hurricane': len(self.filter_by_hazard('hurricane')),
            'flood': len(self.filter_by_hazard('flood')),
            'total_components': sum(len(obj.search_dict) for obj in self.search_objects)
        }
        return summary