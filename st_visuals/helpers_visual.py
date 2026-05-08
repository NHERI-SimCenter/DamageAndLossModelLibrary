import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional
from pelicun.base import convert_to_MultiIndex

@st.cache_data(show_spinner=False)
def load_consequence_df(json_path: str) -> Optional[pd.DataFrame]:
    """
    Load and cache the consequence_repair.csv located alongside fragility.json.

    Applies pelicun's double convert_to_MultiIndex so rows are indexed by
    (comp_id, consequence_type) and columns by (DS label, parameter).

    Returns None if the file does not exist or cannot be parsed.
    """
    cons_csv = Path(json_path).parent / "consequence_repair.csv"
    if not cons_csv.exists():
        return None
    try:
        return convert_to_MultiIndex(
            convert_to_MultiIndex(pd.read_csv(cons_csv, index_col=0), axis=1), axis=0
        )
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_fragility_df(json_path: str) -> Optional[pd.DataFrame]:
    """
    Load and cache the fragility.csv located alongside fragility.json.

    Applies pelicun's convert_to_MultiIndex so columns are indexed by
    (level, parameter) — e.g. ('Demand', 'Type'), ('LS1', 'Theta_0').

    Returns None if the file does not exist or cannot be parsed.
    """
    csv_path = Path(json_path).parent / "fragility.csv"
    if not csv_path.exists():
        return None
    try:
        return convert_to_MultiIndex(
            pd.read_csv(csv_path, index_col=0), axis=1
        )
    except Exception:
        return None