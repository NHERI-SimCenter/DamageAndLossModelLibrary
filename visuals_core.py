"""
Pure (side‑effect‑free) figure builders for fragility & repair functions.

These functions are **framework‑agnostic**: they take a slice of the
component database (plus optional metadata) and return a ready‑to‑render
`plotly.graph_objects.Figure`.  No files are written, no globals mutated.
"""

from __future__ import annotations

from textwrap import wrap
from typing import Any, Dict, Sequence

import colorlover as cl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, weibull_min


# ---------------------------------------------------------------------
# Helper utilities – all internal, no side‑effects
# ---------------------------------------------------------------------

def _sequential_colors(base: str, n: int) -> Sequence[str]:
    """
    Pick *n* colors from a ColorBrewer sequential palette (`Reds`, `PuBu`, …).
    Falls back gracefully if *n* > palette length.
    """
    palette = cl.scales[str(max(3, min(9, n)))]['seq'][base]
    if n <= len(palette):
        return palette[-n:]
    # repeat / cycle if more colors requested than available
    # palettes retrieved from colorlover retrieve at most 9 colors
    return [palette[i % len(palette)] for i in range(n)]


def _wrap_html(text: str, width: int = 70) -> str:
    """Simple word‑wrap helper used for hover‑text annotations."""
    return '<br>'.join(wrap(text, width=width))


# ---------------------------------------------------------------------
# Public API – pure builders
# ---------------------------------------------------------------------

def build_fragility_figure(
    comp_data: pd.Series,
    meta: Dict[str, Any] | None = None,
) -> go.Figure:
    """
    Build the fragility figure for *one* component.

    Parameters
    ----------
    comp_data
        A **row** of the fragility CSV already converted to a
        pandas Series with a two‑level column MultiIndex.
    meta
        Optional metadata dictionary extracted from `<csv>.json`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{'type': 'xy'}, {'type': 'table'}]],
        column_widths=[0.4, 0.6],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # -----------------------------------------------------------------
    # 1. Curves
    # -----------------------------------------------------------------

    #limit_states = list of 
    limit_states = [ls for ls in comp_data.index.unique(level=0) if 'LS' in ls]
    if comp_data.loc[('Incomplete', '')] == 1:
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode='lines',
                line={'width': 3, 'color': _sequential_colors('Reds', 1)[0]},
                name='Incomplete Fragility Data',
            ),
            row=1, col=1,
        )
    else:
        # Choose colors
        colors = _sequential_colors('Reds', len(limit_states))

        # Determine common demand range that covers all CDFs nicely
        #probability min & demand min
        p_min, p_max = 0.01, 0.9
        d_min, d_max = np.inf, -np.inf
        for ls in limit_states:
            fam = comp_data.loc[(ls, 'Family')]
            if fam in ('normal', 'lognormal', 'weibull'):
                theta0, theta1 = comp_data.loc[(ls, 'Theta_0')], comp_data.loc[(ls, 'Theta_1')]
                if fam == 'normal':
                    d_min_i, d_max_i = norm.ppf([p_min, p_max], loc=theta0, scale=theta1 * theta0)
                elif fam == 'lognormal':
                    d_min_i, d_max_i = np.exp(norm.ppf([p_min, p_max], loc=np.log(theta0), scale=theta1))
                else:  # weibull
                    d_min_i, d_max_i = weibull_min.ppf([p_min, p_max], theta1, scale=theta0)
            elif fam == 'multilinear_CDF':
                xs = list(map(float, comp_data.loc[(ls, 'Theta_0')].split('|')[0].split(',')))
                d_min_i, d_max_i = xs[0], xs[-1]
            else:
                continue
            d_min, d_max = min(d_min, d_min_i), max(d_max, d_max_i)

        demand_vals = np.linspace(d_min, d_max, 100)

        for i, ls in enumerate(limit_states):
            fam = comp_data.loc[(ls, 'Family')]
            theta0 = comp_data.loc[(ls, 'Theta_0')]
            theta1 = comp_data.get((ls, 'Theta_1'), np.nan)

            if fam == 'normal':
                cdf = norm.cdf(demand_vals, loc=theta0, scale=theta1 * theta0)
            elif fam == 'lognormal':
                cdf = norm.cdf(np.log(demand_vals), loc=np.log(theta0), scale=theta1)
            elif fam == 'weibull':
                cdf = weibull_min.cdf(demand_vals, theta1, scale=theta0)
            elif fam == 'multilinear_CDF':
                xs, ys = (np.asarray(p.split(','), dtype=float)
                          for p in theta0.split('|'))
                cdf = np.interp(demand_vals, xs, ys)
            else:
                continue

            fig.add_trace(
                go.Scatter(
                    x=demand_vals,
                    y=cdf,
                    mode='lines',
                    line={'width': 3, 'color': colors[i]},
                    name=ls,
                ),
                row=1, col=1,
            )

    # -----------------------------------------------------------------
    # 2. Parameter table (skip for multilinear CDFs)
    # -----------------------------------------------------------------
    if not any(comp_data.loc[(ls, 'Family')] == 'multilinear_CDF' for ls in limit_states):
        table_rows: list[list[Any]] = []
        for ls in limit_states:
            fam = comp_data.loc[(ls, 'Family')]
            if pd.isna(fam):
                continue
            theta0, theta1 = comp_data.loc[(ls, 'Theta_0')], comp_data.get((ls, 'Theta_1'), 'N/A')

            # Special handling for Weibull: present median & dispersion
            if fam == 'weibull':
                lam, kappa = theta0, theta1
                theta0 = round(weibull_min.median(kappa, scale=lam), 2)
                theta1 = round(weibull_min.std(kappa, scale=lam) /
                               weibull_min.mean(kappa, scale=lam), 2)

            table_rows.append([ls, theta0, fam, theta1,
                               comp_data.get((ls, 'DemageStateWeights'), None)])

        # Build DS names column
        ds_names = []
        ds_index = 1
        for weights in [row[-1] for row in table_rows]:
            if pd.isna(weights):
                ds_names.append(f'DS{ds_index}')
                ds_index += 1
            else:
                w = weights.split('|')
                ds_names.append('<br>'.join(f'DS{ds_index + i} ({100*float(val):.0f}%)'
                                            for i, val in enumerate(w)))
                ds_index += len(w)

        # Assemble columns expected by go.Table
        cols = list(zip(*table_rows))
        # Insert DS names as second column and shift others
        cols = [cols[0], ds_names, cols[1], cols[2], cols[3]]

        fig.add_trace(
            go.Table(
                columnwidth=[50, 70, 65, 95, 80],
                header=dict(
                    values=['<b>Limit<br>State</b>',
                            '<b>Damage State(s)</b>',
                            '<b>Median<br>Capacity</b>',
                            '<b>Capacity<br>Distribution</b>',
                            '<b>Capacity<br>Dispersion</b>'],
                    align=['center', 'left', 'center', 'center', 'center'],
                    fill_color="#1B1B1B",
                    fill={'color': 'rgb(200,200,200)'},
                    font={'size': 16},
                ),
                cells=dict(
                    values=cols,
                    align=['center', 'left', 'center', 'center', 'center'],
                    fill={'color': 'rgba(0,0,0,0)'},
                    font={'size': 12},
                ),
            ),
            row=1, col=2,
        )

        # Optional * annotations (hover pop‑ups) from metadata
        if meta:
            x_loc, y_loc = 0.4928, 0.82
            for i, ds_html in enumerate(ds_names):
                ls_meta = meta.get('LimitStates', {}).get(f'LS{i+1}', {})
                if not ls_meta:
                    continue
                for j, ds_id in enumerate(ls_meta.keys()):
                    desc = ls_meta[ds_id].get('Description', '')
                    repair = ls_meta[ds_id].get('RepairAction', '')
                    hover = f"<b>{ds_id}</b><br>{_wrap_html(desc)}"
                    if repair:
                        hover += f"<br><br><b>Repair Action</b><br>{_wrap_html(repair)}"
                    fig.add_annotation(
                        text='<b>*</b>',
                        hovertext=hover,
                        xref='paper', yref='paper',
                        showarrow=False,
                        x=x_loc, y=y_loc,
                        font={'size': 9},
                    )
                    y_loc -= 0.055

    # -----------------------------------------------------------------
    # 3. Axes & layout
    # -----------------------------------------------------------------
    demand_unit = comp_data.loc[('Demand', 'Unit')]
    demand_unit = '-' if demand_unit == 'unitless' else demand_unit
    fig.update_xaxes(title_text=f"{comp_data.loc[('Demand', 'Type')]} [{demand_unit}]",
                     showgrid=True, gridcolor='lightgrey', linecolor='black')
    fig.update_yaxes(title_text='P(LS ≥ ls<sub>i</sub>)', range=[0, 1.02],
                     showgrid=True, gridcolor='lightgrey', linecolor='black')
    fig.update_layout(
        height=300, width=950,
        margin={'l': 5, 'r': 5, 't': 5, 'b': 5},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig


def build_repair_figure(
    comp_data: pd.Series,
    dv_type: str,
    meta: Dict[str, Any] | None = None,
) -> go.Figure:
    """
    Build a repair‑consequence figure for one component & DV type.

    *The implementation mirrors the logic in the original* `plot_repair`
    *but stops right before* `fig.write_html(...)`.  (Copy that block
    almost verbatim, or gradually factor pieces into helpers as above.)
    """
    # ✂️  Implementation omitted for brevity – copy from visuals.py,
    #     following exactly the same stripping pattern shown above.
    raise NotImplementedError("Port logic from plot_repair here.")
