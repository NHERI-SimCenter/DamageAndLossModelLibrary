"""
tree_visuals.py
---------------
Hierarchical tree view for the fragility component library.

Renders a four-level collapsible tree inside Streamlit:

  Seismic
  └── Source  (FEMA P-58 / Hazus …)
      └── Component Group  (B - Shell / GF - Geotechnical Failure …)
          └── Sub-Group  (B.10.31 - Steel Columns / GF.H - Horizontal Spreading …)
              └── Component  [detail panel + fragility / consequence charts]

Usage
-----
    from tree_visuals import render_seismic_tree
    render_seismic_tree()                          # auto-loads all seismic data
    render_seismic_tree(seismic_objects=my_list)   # pass pre-filtered objects

Performance notes
-----------------
* st.expander executes ALL child code on every Streamlit re-run, whether the
  expander is open or closed. To avoid building hundreds of Plotly figures and
  nested widget trees on every interaction, component detail panels are guarded
  by session-state flags — content is only rendered for explicitly opened leaves.
* _build_tree and _load_full_json are cached so JSON parsing and prefix-routing
  only run once per process lifetime.
* Plotly figures are cached per component ID so they are not rebuilt on re-runs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import colorlover as cl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pelicun.base import convert_to_MultiIndex
from plotly.subplots import make_subplots
from scipy.stats import norm, weibull_min

from st_search.component_search import FuzzyIndex, SearchObject


# ─── Palette & constants ───────────────────────────────────────────────────────

_DS_COLORS: List[str] = ["#3b82f6", "#f59e0b", "#ef4444", "#7c3aed", "#10b981"]
_CATEGORY_BADGE: Dict[str, str] = {"FEMA": "🔵 FEMA P-58", "HAZUS": "🟠 Hazus"}

# Session-state key that holds the set of expanded component IDs
_EXPANDED_KEY = "tree_expanded_components"

# Consequence type options shown in the selectbox
_C_TYPES: List[str] = ["Cost", "Time", "Carbon", "Energy"]

# PuBu sequential palette keyed by number of damage states — mirrors plot_repair
_PUBU_COLORS: Dict[int, List[str]] = {
    1: [cl.scales["3"]["seq"]["PuBu"][2]],
    2: cl.scales["3"]["seq"]["PuBu"][1:],
    3: cl.scales["4"]["seq"]["PuBu"][1:],
    4: cl.scales["6"]["seq"]["PuBu"][2:],
    5: cl.scales["7"]["seq"]["PuBu"][2:],
    6: cl.scales["7"]["seq"]["PuBu"][1:],
    7: cl.scales["7"]["seq"]["PuBu"],
}


# ─── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_cached_index() -> FuzzyIndex:
    """Cache the FuzzyIndex so JSON parsing only runs once per process."""
    return FuzzyIndex()


@st.cache_data(show_spinner=False)
def _load_full_json(json_path: str) -> dict:
    """
    Load and cache the complete fragility.json for a given path.

    The SearchObject only stores component descriptions; this function
    retrieves the full component record (Comments, LimitStates, etc.)
    so the detail panel and fragility charts can be populated.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def _load_consequence_df(json_path: str) -> Optional[pd.DataFrame]:
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
def _load_consequence_meta(json_path: str) -> Optional[dict]:
    """
    Load and cache consequence_repair.json metadata from the same directory.

    Returns None if the file is missing or unreadable.
    """
    cons_json = Path(json_path).parent / "consequence_repair.json"
    if not cons_json.exists():
        return None
    try:
        with open(cons_json, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _load_fragility_df(json_path: str) -> Optional[pd.DataFrame]:
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


@st.cache_data(show_spinner=False)
def _build_tree(file_paths: tuple[str, ...]) -> Dict[str, dict]:
    """
    Build the nested source → group → sub-group → component tree.

    Accepts a tuple of file paths (hashable for cache_data) so the result
    survives across re-runs without re-routing all component IDs.

    Returns
    -------
    dict
        {
          short_name: {
            "file_path": str,
            "meta": dict,           # _GeneralInformation
            "groups": {
              group_name: {
                "subgroups": {
                  subgroup_name: {
                    "components": [comp_id, ...]
                  }
                }
              }
            }
          }
        }

    Routing logic
    ~~~~~~~~~~~~~
    Each component ID is matched to a sub-group by longest common prefix.
    Example: "GF.H.S" → prefix "GF.H" → "GF.H - Horizontal Spreading".
    Components that match no defined prefix fall into an (Unclassified) bucket.
    """
    tree: Dict[str, dict] = {}

    for fp in file_paths:
        try:
            data = _load_full_json(fp)
        except Exception:
            continue

        meta: dict = data.get("_GeneralInformation", {})
        short_name: str = meta.get("ShortName", Path(fp).parent.name)

        # Build prefix → label maps from ComponentGroups.
        #
        # ComponentGroups is a dict[str, list[str]]:
        #   { "GF": ["GF.H", "GF.V", "GF.L"],
        #     "STR": ["STR.W1", "STR.S1", ...], ... }
        #
        # Keys are top-level group prefixes; values are lists of subgroup
        # prefixes.  We build two independent maps:
        #   group_map:    top-prefix  → group display label  (e.g. "GF")
        #   subgroup_map: sub-prefix  → subgroup display label (e.g. "GF.H")
        # Both maps store just the prefix as the label because the JSON does
        # not separately provide human-readable group names here.
        raw_cg = meta.get("ComponentGroups", {})
        if not isinstance(raw_cg, dict):
            raw_cg = {}

        group_map: Dict[str, str] = {grp: grp for grp in raw_cg}
        subgroup_map: Dict[str, str] = {
            sg: sg
            for sg_list in raw_cg.values()
            if isinstance(sg_list, list)
            for sg in sg_list
        }

        # Collect all real component IDs (skip keys starting with "_")
        comp_ids = [k for k in data if not k.startswith("_")]

        # Route each component into group -> subgroup buckets.
        # Top-level group: first dot-segment of comp_id (e.g. "GF", "STR").
        # Sub-group: first two dot-segments joined (e.g. "GF.H", "STR.W1").
        # When a prefix has no match in the maps (happens for FEMA P-58 IDs
        # whose sub-groups may not be listed), fall back to the bare segment.
        groups: Dict[str, dict] = {}
        for comp_id in comp_ids:
            parts = comp_id.split(".")
            top_segment = parts[0]
            sub_segment = ".".join(parts[:2]) if len(parts) >= 2 else top_segment

            group_label = group_map.get(top_segment, top_segment)
            subgroup_label = subgroup_map.get(sub_segment, sub_segment)

            groups.setdefault(group_label, {"subgroups": {}})
            groups[group_label]["subgroups"].setdefault(
                subgroup_label, {"components": []}
            )
            groups[group_label]["subgroups"][subgroup_label]["components"].append(
                comp_id
            )

        # Pre-sort component lists and cache per-group counts so the render
        # loop never has to sort or count on re-runs.
        total_count = 0
        for g_data in groups.values():
            g_count = 0
            for sg_data in g_data["subgroups"].values():
                sg_data["components"].sort()
                g_count += len(sg_data["components"])
            g_data["count"] = g_count
            total_count += g_count

        tree[short_name] = {
            "file_path": fp,
            "meta": meta,
            "groups": groups,
            "count": total_count,
        }

    return tree


def _count_components(groups: Dict[str, dict]) -> int:
    """Return the total component count across all groups and sub-groups."""
    return sum(
        len(sg["components"])
        for g in groups.values()
        for sg in g["subgroups"].values()
    )


# ─── Plotly helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _make_fragility_figure(
    comp_id: str,
    limit_states_json: str,
    csv_row_json: str,
) -> go.Figure:
    """
    Build a fragility figure for a single component.

    Parameters
    ----------
    comp_id : str
        Component identifier.
    limit_states_json : str
        JSON-serialised LimitStates dict from fragility.json (used for
        damage-state descriptions in trace names).
    csv_row_json : str
        JSON-serialised pandas Series (from fragility.csv with MultiIndex
        columns) containing the distribution parameters and demand info.
    """
    limit_states: dict = json.loads(limit_states_json)
    csv_row: dict = json.loads(csv_row_json)

    def _isna(v) -> bool:
        """Check for NaN in both float and string forms (JSON roundtrip)."""
        if v is None:
            return True
        if isinstance(v, str) and v.lower() == "nan":
            return True
        try:
            return pd.isna(v)
        except (TypeError, ValueError):
            return False

    # ── Extract demand type and unit ──────────────────────────────────────
    demand_type = csv_row.get("Demand-Type", "Peak Ground Acceleration")
    demand_unit = csv_row.get("Demand-Unit", "g")
    if demand_unit == "unitless":
        demand_unit = "-"

    # ── Build DS description lookup from JSON LimitStates ─────────────────
    ds_descriptions: Dict[str, str] = {}
    for ls_data in limit_states.values():
        for ds_key, ds_data in ls_data.items():
            desc = (
                ds_data.get("Description", "")
                if isinstance(ds_data, dict)
                else str(ds_data)
            )
            ds_descriptions[ds_key] = desc[:50]

    # ── Collect limit states from CSV row ─────────────────────────────────
    ls_keys = sorted(
        {k.split("-")[0] for k in csv_row if k.startswith("LS") and "-" in k}
    )

    # ── Determine demand range ────────────────────────────────────────────
    p_min, p_max = 0.01, 0.9
    d_min, d_max = np.inf, -np.inf
    for ls in ls_keys:
        fam = csv_row.get(f"{ls}-Family")
        theta0 = csv_row.get(f"{ls}-Theta_0")
        theta1 = csv_row.get(f"{ls}-Theta_1")
        if _isna(fam) or _isna(theta0):
            continue
        try:
            theta0_f = float(theta0) if not isinstance(theta0, str) or "|" not in str(theta0) else None
            theta1_f = float(theta1) if theta1 is not None and not _isna(theta1) else None
        except (ValueError, TypeError):
            theta0_f = theta1_f = None

        if fam == "lognormal" and theta0_f and theta1_f:
            d_min_i, d_max_i = np.exp(
                norm.ppf([p_min, p_max], loc=np.log(theta0_f), scale=theta1_f)
            )
        elif fam == "normal" and theta0_f and theta1_f:
            d_min_i, d_max_i = norm.ppf(
                [p_min, p_max], loc=theta0_f, scale=theta1_f * theta0_f
            )
        elif fam == "weibull" and theta0_f and theta1_f:
            d_min_i, d_max_i = weibull_min.ppf(
                [p_min, p_max], theta1_f, scale=theta0_f
            )
        elif fam == "multilinear_CDF" and isinstance(theta0, str):
            xs = list(map(float, theta0.split("|")[0].split(",")))
            d_min_i, d_max_i = xs[0], xs[-1]
        else:
            continue
        d_min, d_max = min(d_min, d_min_i), max(d_max, d_max_i)

    if d_min >= d_max:
        d_min, d_max = 0.0, 1.0
    demand_vals = np.linspace(d_min, d_max, 300)

    # ── Build curves ──────────────────────────────────────────────────────
    fig = go.Figure()
    trace_i = 0
    ds_index = 1

    for ls in ls_keys:
        fam = csv_row.get(f"{ls}-Family")
        theta0 = csv_row.get(f"{ls}-Theta_0")
        theta1 = csv_row.get(f"{ls}-Theta_1")
        weights_str = csv_row.get(f"{ls}-DamageStateWeights")

        if _isna(fam) or _isna(theta0):
            continue

        # Compute CDF
        try:
            if fam == "lognormal":
                cdf = norm.cdf(
                    np.log(demand_vals),
                    loc=np.log(float(theta0)),
                    scale=float(theta1),
                )
            elif fam == "normal":
                t0, t1 = float(theta0), float(theta1)
                cdf = norm.cdf(demand_vals, loc=t0, scale=t1 * t0)
            elif fam == "weibull":
                cdf = weibull_min.cdf(
                    demand_vals, float(theta1), scale=float(theta0)
                )
            elif fam == "multilinear_CDF" and isinstance(theta0, str):
                xs, ys = (
                    np.asarray(p.split(","), dtype=float)
                    for p in theta0.split("|")
                )
                cdf = np.interp(demand_vals, xs, ys)
            else:
                continue
        except (ValueError, TypeError):
            continue

        # Determine DS labels for this limit state
        n_ds = 1
        if weights_str is not None and not _isna(weights_str):
            n_ds = len(str(weights_str).split("|"))

        ds_label_parts = []
        for j in range(n_ds):
            ds_key = f"DS{ds_index + j}"
            desc = ds_descriptions.get(ds_key, "")
            ds_label_parts.append(f"{ds_key}: {desc}" if desc else ds_key)

        trace_name = "; ".join(ds_label_parts)

        fig.add_trace(
            go.Scatter(
                x=demand_vals,
                y=cdf,
                name=trace_name,
                mode="lines",
                line=dict(
                    color=_DS_COLORS[trace_i % len(_DS_COLORS)], width=2.5
                ),
                hovertemplate=(
                    f"<b>{ls}</b><br>"
                    f"{demand_type}: %{{x:.3f}} {demand_unit}<br>"
                    f"P(DS ≥ ds): %{{y:.1%}}<extra></extra>"
                ),
            )
        )
        trace_i += 1
        ds_index += n_ds

    fig.update_layout(
        xaxis_title=f"{demand_type} [{demand_unit}]",
        yaxis=dict(title="P(DS ≥ ds | IM)", tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", y=-0.38, font=dict(size=10)),
        height=340,
        margin=dict(l=60, r=20, t=36, b=130),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(show_spinner=False)
def _make_consequence_figure(
    comp_id: str,
    c_type: str,
    json_path: str,
) -> go.Figure:
    """
    Build a Plotly consequence figure for a single component + consequence type.

    Mirrors the curve panel of pelicun's ``plot_repair()`` for inline rendering
    inside the tree_visuals detail panel: plots the median consequence function
    per damage state, dashed uncertainty bands (±1σ), and a normalised PDF at
    the maximum quantity value.

    Parameters
    ----------
    comp_id : str
        Component ID (e.g. ``"B.10.31.001a"``).
    c_type : str
        Consequence type: one of ``"Cost"``, ``"Time"``, ``"Carbon"``,
        ``"Energy"``.
    json_path : str
        Path to the source ``fragility.json``.  Used to locate
        ``consequence_repair.csv`` (and its metadata JSON) in the same
        directory.

    Returns
    -------
    go.Figure
        Two-column subplot: [consequence curve | end-point PDF].
    """
    # ── Load data ──────────────────────────────────────────────────────────
    repair_df = _load_consequence_df(json_path)

    def _empty_fig(message: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color="#9ca3af"),
            xanchor="center",
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300,
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        return fig

    if repair_df is None:
        return _empty_fig("consequence_repair.csv not found in this source directory")

    lvl0 = repair_df.index.get_level_values(0)
    if comp_id not in lvl0:
        return _empty_fig(f"No consequence data for {comp_id}")

    comp_rows = repair_df.loc[comp_id]
    if c_type not in comp_rows.index:
        return _empty_fig(f"No {c_type} consequence data for {comp_id}")

    comp_data = comp_rows.loc[c_type]

    # ── Guard against incomplete data ─────────────────────────────────────
    if comp_data.loc[("Incomplete", "")] == 1:
        return _empty_fig(f"Incomplete {c_type} consequence data for {comp_id}")

    # ── Damage state model parameters ─────────────────────────────────────
    limit_states = [v for v in comp_data.index.unique(level=0) if "DS" in v]

    table_vals: list = []
    for ls in limit_states:
        fields = ["Theta_0", "Family", "Theta_1"]
        ds_row = comp_data[ls].copy()
        for opt in ["Family", "Theta_1"]:
            if opt not in ds_row.index:
                ds_row[opt] = None
        if not np.all(pd.isna(ds_row[fields].values)):
            table_vals.append(np.insert(ds_row[fields].values, 0, ls))

    if not table_vals:
        return _empty_fig(f"No {c_type} model parameters available for {comp_id}")

    # model_params rows: [0] DS label, [1] Theta_0, [2] Family, [3] Theta_1
    model_params = np.array(table_vals).T
    n_ds = model_params.shape[1]

    # ── Quantity axis limits ───────────────────────────────────────────────
    q_min, q_max = 0.0, -np.inf
    for mu in model_params[1]:
        if "|" in str(mu):
            q_lims = np.array(mu.split("|")[1].split(","), dtype=float)
            q_max = np.max([np.sum(q_lims), q_max])
    if q_max == -np.inf:
        q_max = 1.0

    need_x_axis = any("|" in str(mu) for mu in model_params[1])

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.85, 0.15],
        horizontal_spacing=0.02,
    )

    color_key = min(n_ds, 7)
    palette = _PUBU_COLORS[color_key]

    # ── Per-DS plotting (mirrors plot_repair curve logic) ─────────────────
    for ds_i, mu_capacity in enumerate(model_params[1]):
        ds_label = model_params[0][ds_i]
        ds_color = palette[ds_i % color_key]

        # Median consequence function ──────────────────────────────────────
        if "|" in str(mu_capacity):
            c_vals, q_vals = np.array(
                [v.split(",") for v in mu_capacity.split("|")], dtype=float
            )
        else:
            c_vals = np.array([mu_capacity], dtype=float)
            q_vals = np.array([0.0], dtype=float)

        # Extend both ends to cover the full quantity axis
        q_vals = np.insert(q_vals, 0, q_min)
        c_vals = np.insert(c_vals, 0, c_vals[0])
        q_vals = np.append(q_vals, q_max)
        c_vals = np.append(c_vals, c_vals[-1])

        fig.add_trace(
            go.Scatter(
                x=q_vals, y=c_vals,
                mode="lines",
                line=dict(width=3, color=ds_color),
                name=ds_label,
                legendgroup=ds_label,
            ),
            row=1, col=1,
        )

        # Uncertainty bands ────────────────────────────────────────────────
        dispersion = model_params[3][ds_i]
        if pd.isna(dispersion) or dispersion == "N/A":
            continue

        dispersion = float(dispersion)
        dist = model_params[2][ds_i]

        if dist == "normal":
            std_plus = c_vals * (1.0 + dispersion)
            std_minus = np.maximum(c_vals * (1.0 - dispersion), 0.0)
            lbl_plus, lbl_minus = "mu + std", "mu - std"
        elif dist == "lognormal":
            std_plus = np.exp(np.log(c_vals) + dispersion)
            std_minus = np.exp(np.log(c_vals) - dispersion)
            lbl_plus, lbl_minus = "mu + lnstd", "mu - lnstd"
        else:
            continue

        for y_band, band_label in [(std_plus, lbl_plus), (std_minus, lbl_minus)]:
            fig.add_trace(
                go.Scatter(
                    x=q_vals, y=y_band,
                    mode="lines",
                    line=dict(width=1, color=ds_color, dash="dash"),
                    name=f"{ds_label} {band_label}",
                    legendgroup=ds_label,
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # End-point PDF ────────────────────────────────────────────────────
        c_end = float(c_vals[-1])
        if c_end <= 0:
            continue

        if dist == "normal":
            sig = c_end * dispersion
            q_pdf = np.linspace(
                np.max([norm.ppf(0.025, loc=c_end, scale=sig), 0.0]),
                norm.ppf(0.975, loc=c_end, scale=sig),
                num=100,
            )
            c_pdf = norm.pdf(q_pdf, loc=c_end, scale=sig)

        elif dist == "lognormal":
            q_pdf = np.linspace(
                np.exp(norm.ppf(0.025, loc=np.log(c_end), scale=dispersion)),
                np.exp(norm.ppf(0.975, loc=np.log(c_end), scale=dispersion)),
                num=100,
            )
            c_pdf = norm.pdf(np.log(q_pdf), loc=np.log(c_end), scale=dispersion)

        c_pdf = c_pdf / np.max(c_pdf)

        fig.add_trace(
            go.Scatter(
                x=c_pdf, y=q_pdf,
                mode="lines",
                line=dict(width=1, color=ds_color),
                fill="tozeroy",
                name=f"{ds_label} pdf",
                legendgroup=ds_label,
                showlegend=False,
            ),
            row=1, col=2,
        )

    # ── Axis labels from DB units ─────────────────────────────────────────
    quantity_unit: str = comp_data.loc[("Quantity", "Unit")]
    if quantity_unit in ("unitless", "1 EA", "1 ea"):
        quantity_unit = "-"
    elif quantity_unit.split()[0] == "1":
        quantity_unit = quantity_unit.split()[1]

    dv_unit: str = comp_data.loc[("DV", "Unit")]
    if dv_unit == "unitless":
        dv_unit = "-"

    shared_ax = dict(
        showgrid=True,
        linecolor="black",
        gridwidth=0.05,
        gridcolor="rgb(220,220,220)",
    )

    fig.update_layout(
        margin=dict(b=50, r=5, l=5, t=30),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        xaxis1=(
            dict(title_text=f"Damage Quantity [{quantity_unit}]", **shared_ax)
            if need_x_axis
            else dict(showgrid=False, showticklabels=False)
        ),
        yaxis1=dict(
            title_text=f"{c_type} [{dv_unit}]",
            rangemode="tozero",
            **shared_ax,
        ),
        xaxis2=dict(showgrid=False, showticklabels=False, title_text=""),
        yaxis2=dict(showgrid=False, showticklabels=False),
        legend=dict(
            yanchor="top",
            xanchor="right",
            font=dict(size=11),
            orientation="v",
            y=1.0,
            x=-0.08,
        ),
        template="plotly_white",
    )
    return fig


# ─── Component detail panel ────────────────────────────────────────────────────

def _render_component_detail(
    comp_id: str,
    comp_data: dict,
    json_path: str,
) -> None:
    """
    Render the inline detail panel for a leaf component node.

    Only called when the user has explicitly opened the component — never
    rendered unconditionally inside a collapsed expander.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"B.10.31.001a"``.
    comp_data : dict
        Full component record loaded from fragility.json.
    json_path : str
        Path to the source fragility.json.  Passed through to consequence
        helpers so they can locate ``consequence_repair.csv`` in the same
        directory.
    """
    description = comp_data.get("Description", "")
    comments = comp_data.get("Comments", "")
    block_size = comp_data.get("SuggestedComponentBlockSize", "")
    round_up = comp_data.get("RoundUpToIntegerQuantity", "")
    limit_states: dict = comp_data.get("LimitStates", {})

    col_left, col_right = st.columns([1, 2], gap="large")

    # ── Left: metadata + damage states ────────────────────────────────────
    with col_left:
        st.markdown("**Component metadata**")

        for label, value in [
            ("ID", f"`{comp_id}`"),
            ("Block size", f"`{block_size}`" if block_size else "—"),
            ("Integer qty", round_up if round_up else "—"),
        ]:
            c1, c2 = st.columns([1, 1])
            c1.caption(label)
            c2.caption(value)

        if description:
            st.caption(f"_{description}_")

        st.divider()
        st.markdown("**Damage states**")

        total_ds = sum(len(ds_dict) for ds_dict in limit_states.values())
        if total_ds:
            for ls_key, ls_data in limit_states.items():
                for ds_key, ds_data in ls_data.items():
                    desc_text = (
                        ds_data.get("Description", "No description.")
                        if isinstance(ds_data, dict)
                        else str(ds_data)
                    )
                    with st.expander(f"{ls_key} / {ds_key}", expanded=False):
                        st.caption(desc_text)
                        if isinstance(ds_data, dict) and ds_data.get("RepairAction"):
                            st.caption(f"**Repair action:** {ds_data['RepairAction']}")
        else:
            st.caption("No limit-state data found.")

    # ── Right: comments + charts ───────────────────────────────────────────
    with col_right:
        if comments:
            with st.expander("Technical notes / comments", expanded=False):
                st.caption(comments)

        tab_frag, tab_cons = st.tabs(["Fragility curves", "Consequence curves"])

        with tab_frag:
            frag_df = _load_fragility_df(json_path)
            if frag_df is not None and comp_id in frag_df.index:
                csv_row = frag_df.loc[comp_id]
                # Flatten MultiIndex columns to "Level-Param" keys for JSON
                csv_row_flat = {
                    f"{a}-{b}" if b else str(a): v
                    for (a, b), v in csv_row.items()
                }
                st.plotly_chart(
                    _make_fragility_figure(
                        comp_id,
                        json.dumps(limit_states),
                        json.dumps(csv_row_flat, default=str),
                    ),
                    use_container_width=True,
                    key=f"frag_{comp_id}",
                )
            else:
                st.info("No fragility data available to generate curves.", icon="ℹ️")

        with tab_cons:
            _render_consequence_tab(comp_id, json_path)


def _render_consequence_tab(comp_id: str, json_path: str) -> None:
    """
    Render the consequence curves tab content.

    Checks which consequence types are available for the component and
    lets the user select one via a radio widget before plotting.  Falls
    back to an informational message when no data is found.
    """
    repair_df = _load_consequence_df(json_path)

    if repair_df is None:
        st.info(
            "No consequence data file found for this source directory.",
            icon="ℹ️",
        )
        return

    lvl0 = repair_df.index.get_level_values(0)
    if comp_id not in lvl0:
        st.info(
            f"No consequence records found for `{comp_id}` in the repair database.",
            icon="ℹ️",
        )
        return

    # Discover which consequence types are present for this component
    available_types = [
        t for t in _C_TYPES if t in repair_df.loc[comp_id].index
    ]

    if not available_types:
        st.info("No consequence types available for this component.", icon="ℹ️")
        return

    # Consequence type selector
    c_type = st.radio(
        "Consequence type",
        options=available_types,
        horizontal=True,
        key=f"cons_type_{comp_id}",
    )

    st.plotly_chart(
        _make_consequence_figure(comp_id, c_type, json_path),
        use_container_width=True,
        key=f"cons_{comp_id}_{c_type}",
    )

    # Metadata annotations (description / repair action per DS)
    # cons_meta = _load_consequence_meta(json_path)
    # if cons_meta and comp_id in cons_meta:
    #     ds_meta: dict = cons_meta[comp_id].get("DamageStates", {})
    #     if ds_meta:
    #         with st.expander("Damage state descriptions", expanded=False):
    #             for ds_key, ds_info in ds_meta.items():
    #                 if not isinstance(ds_info, dict):
    #                     continue
    #                 ds_desc = ds_info.get("Description", "")
    #                 ds_repair = ds_info.get("RepairAction", "")
    #                 lines = [f"**{ds_key}**"]
    #                 if ds_desc:
    #                     lines.append(ds_desc)
    #                 if ds_repair:
    #                     lines.append(f"*Repair action:* {ds_repair}")
    #                 st.caption("  \n".join(lines))
    #                 st.divider()


# ─── Tree renderer ─────────────────────────────────────────────────────────────

def render_seismic_tree(
    seismic_objects: Optional[List[SearchObject]] = None,
) -> None:
    """
    Render the seismic component library as a four-level collapsible tree.

    Parameters
    ----------
    seismic_objects : list of SearchObject, optional
        Pre-filtered list of seismic SearchObjects.  When None, all seismic
        objects are loaded from the cached FuzzyIndex.

    Performance strategy
    --------------------
    * The FuzzyIndex, _build_tree result, and all Plotly figures are cached
      at the process level — they survive re-runs without re-computation.
    * Streamlit executes expander child code on every re-run even when
      collapsed, so component detail panels (_render_component_detail) are
      guarded by a session-state set. Detail content is only rendered for
      components the user has explicitly opened, keeping the widget tree
      small on every run.
    """
    # ── Session state for tracking which components are open ───────────────
    if _EXPANDED_KEY not in st.session_state:
        st.session_state[_EXPANDED_KEY] = set()

    # ── Load data ──────────────────────────────────────────────────────────
    if seismic_objects is None:
        with st.spinner("Loading seismic fragility index…"):
            seismic_objects = _get_cached_index().filter_by_hazard("seismic")

    if not seismic_objects:
        st.warning(
            "No seismic fragility data found. Check directory structure.",
            icon="⚠️",
        )
        return

    # Deduplicate file paths and pass a hashable tuple to the cached builder
    file_paths: tuple[str, ...] = tuple(
        dict.fromkeys(obj.file_path for obj in seismic_objects if obj.file_path)
    )
    tree = _build_tree(file_paths)

    # Build a file_path → SearchObject lookup to avoid O(N) scans per source.
    obj_by_path: Dict[str, SearchObject] = {
        o.file_path: o for o in seismic_objects if o.file_path
    }

    # Counts are stored in the tree dict by _build_tree — no re-computation needed.
    total = sum(src["count"] for src in tree.values())

    # ══ Root header ══════════════════════════════════════════════════════════
    st.markdown("## 🌍 Seismic")
    st.caption(
        f"{len(tree)} source{'s' if len(tree) != 1 else ''} · {total:,} components"
    )
    st.divider()

    for short_name, source_data in tree.items():
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]
        groups: Dict[str, dict] = source_data["groups"]

        obj = obj_by_path.get(fp)
        badge = _CATEGORY_BADGE.get(getattr(obj, "category", ""), "")
        n_comp = source_data["count"]

        # ══ Level 2: Source ══════════════════════════════════════════════════
        with st.expander(
            f"**{short_name}**  ·  {badge}  ·  `{n_comp:,}` components",
            expanded=False,
        ):
            if meta.get("Description"):
                st.caption(meta["Description"])

            version = meta.get("Version", "")
            fname = Path(fp).name if fp else "unknown"
            st.caption(f"Version: {version}  ·  File: `{fname}`")
            st.divider()

            for group_name, group_data in groups.items():
                group_total = group_data["count"]
                if group_total == 0:
                    continue

                # ══ Level 3: Component group ══════════════════════════════════
                with st.expander(
                    f"**{group_name}**  ·  `{group_total}` components",
                    expanded=False,
                ):
                    for sg_name, sg_data in group_data["subgroups"].items():
                        comps: List[str] = sg_data["components"]  # pre-sorted in _build_tree
                        if not comps:
                            continue

                        n_sg = len(comps)
                        sg_label = (
                            f"**{sg_name}**  ·  "
                            f"`{n_sg}` component{'s' if n_sg != 1 else ''}"
                        )

                        # ══ Level 4: Sub-group ════════════════════════════════
                        with st.expander(sg_label, expanded=False):
                            for comp_id in comps:
                                # comps is pre-sorted by _build_tree; no sort needed.
                                # _load_full_json is cached — O(1) after first call.
                                full_json: dict = _load_full_json(fp)
                                comp_data: dict = full_json.get(comp_id, {})
                                raw_desc: str = comp_data.get(
                                    "Description",
                                    obj.search_dict.get(comp_id, "") if obj else "",
                                )
                                preview = (
                                    raw_desc[:90] + "…"
                                    if len(raw_desc) > 90
                                    else raw_desc
                                )

                                # ══ Level 5: Component leaf ════════════════════
                                # Session-state guard: detail content is only
                                # rendered after an explicit "Load" click.
                                # Without this guard, Streamlit executes every
                                # expander's body on every re-run (open or not),
                                # so st.tabs / st.plotly_chart / _render_consequence_tab
                                # would fire for ALL components on every interaction.
                                load_key = f"loaded_{comp_id}"
                                with st.expander(
                                    f"🔩  **{comp_id}**  ·  {preview}",
                                    expanded=False,
                                ):
                                    if load_key not in st.session_state:
                                        if st.button(
                                            "Load details",
                                            key=f"btn_{comp_id}",
                                            type="secondary",
                                        ):
                                            st.session_state[load_key] = True
                                            st.rerun()
                                    elif comp_data:
                                        _render_component_detail(
                                            comp_id, comp_data, fp
                                        )
                                    else:
                                        st.warning(
                                            f"Full data for `{comp_id}` was not found "
                                            f"in `{fname}`. The component description "
                                            "is available but detailed fields are missing.",
                                            icon="⚠️",
                                        )

# ─── Wind component detail panel ──────────────────────────────────────────────

def _render_wind_component_detail(
    comp_id: str,
    comp_data: dict,
    json_path: str,
) -> None:
    """
    Render the inline detail panel for a wind library component leaf node.

    Mirrors ``_render_component_detail`` but omits the consequence tab
    because the SimCenter Wind Component Library has no consequence data.

    Parameters
    ----------
    comp_id : str
        Component identifier, e.g. ``"DOOR.garage.001a"``.
    comp_data : dict
        Full component record from fragility.json.
    json_path : str
        Path to the source fragility.json.
    """
    description: str = comp_data.get("Description", "")
    comments: str = comp_data.get("Comments", "")
    references: list = comp_data.get("Reference", [])
    block_size: str = comp_data.get("SuggestedComponentBlockSize", "")
    limit_states: dict = comp_data.get("LimitStates", {})

    col_left, col_right = st.columns([1, 2])

    # ── Left: metadata ─────────────────────────────────────────────────────
    with col_left:
        if description:
            st.markdown(f"**Description:** {description}")

        if block_size:
            st.caption(f"Block size: `{block_size}`")

        if references:
            st.caption("References: " + ", ".join(f"`{r}`" for r in references))

        if limit_states:
            st.markdown("**Limit states / damage states**")
            for ls_key, ls_data in limit_states.items():
                if not isinstance(ls_data, dict):
                    continue
                for ds_key, ds_data in ls_data.items():
                    desc_text = (
                        ds_data.get("Description", "")
                        if isinstance(ds_data, dict)
                        else str(ds_data)
                    )
                    with st.expander(f"{ls_key} / {ds_key}", expanded=False):
                        st.caption(desc_text)
        else:
            st.caption("No limit-state data found.")

    # ── Right: comments + fragility chart ─────────────────────────────────
    with col_right:
        if comments:
            with st.expander("Technical notes / comments", expanded=False):
                st.caption(comments)

        frag_df = _load_fragility_df(json_path)
        if frag_df is not None and comp_id in frag_df.index:
            csv_row = frag_df.loc[comp_id]
            csv_row_flat = {
                f"{a}-{b}" if b else str(a): v
                for (a, b), v in csv_row.items()
            }
            st.plotly_chart(
                _make_fragility_figure(
                    comp_id,
                    json.dumps(limit_states),
                    json.dumps(csv_row_flat, default=str),
                ),
                use_container_width=True,
                key=f"wind_frag_{comp_id}",
            )
        else:
            st.info("No fragility data available to generate curves.", icon="ℹ️")


# ─── Wind tree renderer ────────────────────────────────────────────────────────

# Map top-level component prefixes to human-readable group labels.
# Derived from the IDs present in the SimCenter Wind Component Library.
_WIND_GROUP_LABELS: Dict[str, str] = {
    "DOOR":  "DOOR — Doors",
    "RCOV":  "RCOV — Roof Cover",
    "RSH":   "RSH — Roof Sheathing",
    "RWC":   "RWC — Roof-Wall Connections",
    "WALL":  "WALL — Walls",
    "WCOV":  "WCOV — Wall Cover",
    "WIN":   "WIN — Windows",
    "WSH":   "WSH — Wall Sheathing",
}


def render_wind_tree(
    wind_objects: Optional[List[SearchObject]] = None,
) -> None:
    """
    Render the SimCenter Wind Component Library as a collapsible tree.

    The tree has the same four-level structure used by ``render_seismic_tree``:

      Wind (Hurricane)
      └── Source  (SimCenter Wind Component Library)
          └── Component Group  (DOOR / WIN / RSH …)
              └── Sub-Group  (DOOR.garage / WIN.regular …)
                  └── Component  [detail panel + fragility chart]

    Parameters
    ----------
    wind_objects : list of SearchObject, optional
        Pre-filtered list of wind/hurricane component SearchObjects.
        When ``None``, all hurricane component objects are loaded from
        the cached FuzzyIndex, filtered to the
        ``hurricane/building/component`` path prefix so that Hazus
        *portfolio* sources are excluded.

    Performance strategy
    --------------------
    Identical to ``render_seismic_tree``: FuzzyIndex and _build_tree are
    cached at the process level; component detail panels are guarded by
    session-state flags so they are only rendered after an explicit
    "Load details" click.
    """
    # ── Session state ──────────────────────────────────────────────────────
    if _EXPANDED_KEY not in st.session_state:
        st.session_state[_EXPANDED_KEY] = set()

    # ── Load data ──────────────────────────────────────────────────────────
    if wind_objects is None:
        with st.spinner("Loading wind fragility index…"):
            all_hurricane = _get_cached_index().filter_by_hazard("hurricane")
            # Restrict to component-level sources only (exclude portfolio models
            # such as Hazus v5.1 which live under hurricane/building/portfolio).
            wind_objects = [
                o for o in all_hurricane
                if "/building/component/" in o.file_path
            ]

    if not wind_objects:
        st.warning(
            "No wind component fragility data found. "
            "Check that hurricane/building/component/ exists in the directory structure.",
            icon="⚠️",
        )
        return

    file_paths: tuple[str, ...] = tuple(
        dict.fromkeys(obj.file_path for obj in wind_objects if obj.file_path)
    )
    tree = _build_tree(file_paths)

    obj_by_path: Dict[str, SearchObject] = {
        o.file_path: o for o in wind_objects if o.file_path
    }

    total = sum(src["count"] for src in tree.values())

    # ══ Root header ══════════════════════════════════════════════════════════
    st.markdown("## 🌀 Wind (Hurricane)")
    st.caption(
        f"{len(tree)} source{'s' if len(tree) != 1 else ''} · {total:,} components"
    )
    st.divider()

    for short_name, source_data in tree.items():
        fp: str = source_data["file_path"]
        meta: dict = source_data["meta"]
        groups: Dict[str, dict] = source_data["groups"]

        n_comp = source_data["count"]
        fname = Path(fp).name if fp else "unknown"

        # ══ Level 2: Source ══════════════════════════════════════════════════
        with st.expander(
            f"**{short_name}**  ·  🌐 SimCenter  ·  `{n_comp:,}` components",
            expanded=False,
        ):
            if meta.get("Description"):
                st.caption(meta["Description"])

            version = meta.get("Version", "")
            st.caption(f"Version: {version}  ·  File: `{fname}`")
            st.divider()

            for group_prefix, group_data in groups.items():
                group_total = group_data["count"]
                if group_total == 0:
                    continue

                # Apply human-readable label if available.
                group_label = _WIND_GROUP_LABELS.get(group_prefix, group_prefix)

                # ══ Level 3: Component group ══════════════════════════════════
                with st.expander(
                    f"**{group_label}**  ·  `{group_total}` components",
                    expanded=False,
                ):
                    for sg_name, sg_data in group_data["subgroups"].items():
                        comps: List[str] = sg_data["components"]
                        if not comps:
                            continue

                        n_sg = len(comps)
                        sg_label = (
                            f"**{sg_name}**  ·  "
                            f"`{n_sg}` component{'s' if n_sg != 1 else ''}"
                        )

                        # ══ Level 4: Sub-group ════════════════════════════════
                        with st.expander(sg_label, expanded=False):
                            for comp_id in comps:
                                full_json: dict = _load_full_json(fp)
                                comp_data_entry: dict = full_json.get(comp_id, {})
                                raw_desc: str = comp_data_entry.get("Description", "")
                                preview = (
                                    raw_desc[:90] + "…"
                                    if len(raw_desc) > 90
                                    else raw_desc
                                )

                                # ══ Level 5: Component leaf ════════════════════
                                load_key = f"wind_loaded_{comp_id}"
                                with st.expander(
                                    f"🔩  **{comp_id}**  ·  {preview}",
                                    expanded=False,
                                ):
                                    if load_key not in st.session_state:
                                        if st.button(
                                            "Load details",
                                            key=f"wind_btn_{comp_id}",
                                            type="secondary",
                                        ):
                                            st.session_state[load_key] = True
                                            st.rerun()
                                    elif comp_data_entry:
                                        _render_wind_component_detail(
                                            comp_id, comp_data_entry, fp
                                        )
                                    else:
                                        st.warning(
                                            f"Full data for `{comp_id}` was not found "
                                            f"in `{fname}`.",
                                            icon="⚠️",
                                        )