import json
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional
from pelicun.base import convert_to_MultiIndex


@st.cache_data(show_spinner=False)
def load_full_json(json_path: str) -> dict:
    """
    Load and cache a full fragility.json by path.

    Shared by the tree, the search results, and the added-components list so the
    same file is read from disk at most once per session. Component detail
    panels need the complete record (LimitStates, Comments, …) that the search
    index does not store.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# Consequence CSV filenames, in priority order. Hazus hurricane "coupled" uses
# the damage-state ``consequence_repair.csv``; "original" uses a continuous
# ``loss_repair.csv`` (loss ratio vs wind speed). A directory has at most one.
_CONSEQUENCE_CSV_NAMES = ("consequence_repair.csv", "loss_repair.csv")


@st.cache_data(show_spinner=False)
def load_consequence_df(json_path: str) -> Optional[pd.DataFrame]:
    """
    Load and cache the consequence/loss CSV located alongside *json_path*.

    Looks for ``consequence_repair.csv`` first, then ``loss_repair.csv`` (the
    Hazus hurricane "original" loss-function form). Applies pelicun's double
    convert_to_MultiIndex so rows are indexed by (comp_id, consequence_type)
    and columns by (DS / LossFunction label, parameter).

    Returns None if no consequence CSV exists or it cannot be parsed.
    """
    folder = Path(json_path).parent
    for name in _CONSEQUENCE_CSV_NAMES:
        cons_csv = folder / name
        if cons_csv.exists():
            try:
                return convert_to_MultiIndex(
                    convert_to_MultiIndex(pd.read_csv(cons_csv, index_col=0), axis=1),
                    axis=0,
                )
            except Exception:
                return None
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