import json
import os
import pandas as pd

def extract_search_metadata(json_path: str) -> dict:

    processign_df = pd.read_json(json_path)
    processign_df.pop('_GeneralInformation')
    processign_df = processign_df.loc['Description']
    
    processed_dict = processign_df.to_dict()

    return processed_dict
