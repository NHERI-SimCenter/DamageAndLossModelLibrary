# figures.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List

import pandas as pd
import streamlit as st
from pelicun.base import convert_to_MultiIndex
from visuals_core import build_fragility_figure


# -------- Discovery --------
@st.cache_data(show_spinner=False)
def _discover_fragility_csvs(base_paths: Optional[List[str | Path]] = None) -> list[Path]:
    """
    Find every fragility.csv by first discovering fragility.json and
    then taking the sibling CSV in the same directory.
    Mirrors the intent of parse_all_fragility_json()'s recursive scan.
    """
    if base_paths is None:
        base_paths = [Path.cwd(), Path("/mnt/data")]  # adjust/add project roots as needed

    csv_paths: list[Path] = []
    seen_dirs: set[Path] = set()

    for base in map(Path, base_paths):
        if not base.exists():
            continue
        # Recursive search for fragility.json
        for json_path in base.rglob("fragility.json"):
            d = json_path.parent.resolve()
            if d in seen_dirs:
                continue
            candidate = d / "fragility.csv"
            if candidate.exists():
                csv_paths.append(candidate)
                seen_dirs.add(d)

    return csv_paths


@st.cache_data(show_spinner=False)
def load_all_fragility() -> pd.DataFrame:
    """
    Load and combine all discovered fragility.csv files.
    Adds a 'dataset_dir' column so we can trace the source dataset.
    """
    csvs = _discover_fragility_csvs()
    if not csvs:
        raise FileNotFoundError("No fragility.json/fragility.csv pairs were discovered.")

    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            df["dataset_dir"] = str(p.parent)
            frames.append(df)
        except Exception as e:
            # Don't fail the whole app for one bad file; warn instead.
            st.warning(f"Skipping {p}: {e}")
    if not frames:
        raise RuntimeError("All discovered fragility.csv files failed to load.")
    return pd.concat(frames, ignore_index=True)


# -------- Helpers --------
def _get_row_by_component_id(df: pd.DataFrame, comp_id: str | int) -> Optional[pd.Series]:
    id_cols = [c for c in df.columns if c.lower() in ("id", "component_id", "componentid", "fragility_id")]
    # Try explicit ID columns first
    for col in id_cols:
        hits = df[df[col].astype(str) == str(comp_id)]
        if not hits.empty:
            return hits.iloc[0]
    # If there’s exactly one object column, use it as a fallback
    obj_cols = [c for c in df.columns if df[c].dtype == "O"]
    if len(obj_cols) == 1:
        hits = df[df[obj_cols[0]].astype(str) == str(comp_id)]
        if not hits.empty:
            return hits.iloc[0]
    return None


def _build_fragility_figure_from_row(row: pd.Series):
    mi_row = convert_to_MultiIndex(row, axis=0)
    return build_fragility_figure(mi_row)


# -------- Public API --------
def render_selected_fragility_figures(
    selected_ids: Optional[Iterable[str | int]] = None,
    df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Stack Plotly fragility figures for the selected component IDs.

    If `df` is not provided, this will discover *all* datasets by scanning for
    fragility.json and loading each sibling fragility.csv, then search across all.
    """
    ids = list(selected_ids or st.session_state.get("selected_component_ids", []) or [])
    if not ids:
        st.info("Use the search to select components; figures will appear here.")
        return

    try:
        data = df if df is not None else load_all_fragility()
    except Exception as e:
        st.error(f"Failed to load fragility data: {e}")
        return

    st.subheader("Selected components")
    for comp_id in ids:
        with st.container():
            st.markdown(f"**Component:** `{comp_id}`")
            row = _get_row_by_component_id(data, comp_id)
            if row is None:
                st.warning("No matching row found across discovered fragility datasets.")
                continue
            try:
                fig = _build_fragility_figure_from_row(row)
                # Optional: show which dataset this came from
                src = row.get("dataset_dir", "unknown")
                st.caption(f"Source dataset: {src}")
                st.plotly_chart(fig, width='stretch')
            except Exception as ex:
                st.error(f"Failed to build figure for {comp_id}: {ex}")
