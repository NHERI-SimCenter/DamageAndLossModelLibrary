import json
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorlover as cl


from st_visuals.helpers_visual import load_consequence_df, load_fragility_df


# Palette
_DS_COLORS: List[str] = ["#3b82f6", "#f59e0b", "#ef4444", "#7c3aed", "#10b981"]

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


@st.cache_data(show_spinner=False)
def make_consequence_figure(
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
    repair_df = load_consequence_df(json_path)

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

@st.cache_data(show_spinner=False)
def make_fragility_figure(
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
