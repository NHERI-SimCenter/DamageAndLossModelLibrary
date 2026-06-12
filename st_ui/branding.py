"""
branding.py
-----------
Organization attribution / branding for the app.

Shows the organizations behind the project — the NSF NHERI SimCenter (whose
Damage and Loss Model Library this app is built on) and Degenkolb Engineers
(who built this app). Logos are embedded as base64 data URIs from ``assets/``
so there is no runtime dependency on external URLs, and they sit on a light
card so the dark Degenkolb wordmark stays legible on both light and dark themes.

* ``render_header(title)`` — the page header: title and contributor logos on
  the same line (title left, logos vertically centered on the right).
* ``render_brand_header()`` — just the compact, right-aligned logo bar.
* ``render_contributors()`` — fuller centered footer block.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

_SIMCENTER_URL = "https://simcenter.designsafe-ci.org/"
_DEGENKOLB_URL = "https://degenkolb.com/"


@lru_cache(maxsize=None)
def _data_uri(filename: str, mime: str) -> str:
    """Return a base64 ``data:`` URI for a logo file in ``assets/``."""
    raw = (_ASSETS / filename).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _logos() -> tuple[str, str] | None:
    """(simcenter, degenkolb) data URIs, or None if the assets are missing."""
    try:
        return (
            _data_uri("nsf_nheri_simcenter.png", "image/png"),
            _data_uri("degenkolb.svg", "image/svg+xml"),
        )
    except FileNotFoundError:
        return None


def render_header(title: str) -> None:
    """
    Render the page header — the title and the contributor logos on the same
    line: title on the left, logos vertically centered on the right.

    Grouping the two here (rather than calling the title and logos separately
    from the page) keeps the header layout in one place.
    """
    col_title, col_logos = st.columns([3, 2], vertical_alignment="center")
    with col_title:
        st.title(title)
    with col_logos:
        render_brand_header()


def render_brand_header() -> None:
    """Render a compact, right-aligned logo bar."""
    logos = _logos()
    if logos is None:
        return
    simcenter, degenkolb = logos

    st.markdown(
        f"""
        <div style="display:flex;justify-content:flex-end;align-items:center;
                    gap:0.95rem;margin:0;flex-wrap:wrap;">
          <span style="color:#8a8a8a;font-size:0.7rem;letter-spacing:0.09em;
                       text-transform:uppercase;">Contributors</span>
          <span style="display:inline-flex;align-items:center;gap:1.4rem;
                       background:#ffffff;border:1px solid rgba(128,128,128,0.20);
                       border-radius:10px;padding:0.4rem 1.1rem;
                       box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <a href="{_SIMCENTER_URL}" target="_blank" rel="noopener"
               title="NSF NHERI SimCenter">
              <img src="{simcenter}" alt="NSF NHERI SimCenter"
                   style="height:32px;display:block;">
            </a>
            <a href="{_DEGENKOLB_URL}" target="_blank" rel="noopener"
               title="Degenkolb Engineers">
              <img src="{degenkolb}" alt="Degenkolb Engineers"
                   style="height:24px;display:block;">
            </a>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_contributors() -> None:
    """Render a fuller centered contributor block (footer variant)."""
    logos = _logos()
    if logos is None:
        return
    simcenter, degenkolb = logos

    st.markdown(
        f"""
        <div style="margin-top:2.75rem;">
          <hr style="border:none;border-top:1px solid rgba(128,128,128,0.25);
                     margin-bottom:1.1rem;">
          <p style="text-align:center;color:#8a8a8a;font-size:0.78rem;
                    letter-spacing:0.10em;text-transform:uppercase;
                    margin-bottom:0.9rem;">Contributors</p>
          <div style="display:flex;align-items:center;justify-content:center;
                      gap:2.75rem;flex-wrap:wrap;background:#ffffff;
                      border:1px solid rgba(128,128,128,0.20);border-radius:14px;
                      padding:1.1rem 2.25rem;max-width:640px;margin:0 auto;
                      box-shadow:0 1px 4px rgba(0,0,0,0.06);">
            <a href="{_SIMCENTER_URL}" target="_blank" rel="noopener"
               title="NSF NHERI SimCenter">
              <img src="{simcenter}" alt="NSF NHERI SimCenter"
                   style="height:54px;display:block;">
            </a>
            <a href="{_DEGENKOLB_URL}" target="_blank" rel="noopener"
               title="Degenkolb Engineers">
              <img src="{degenkolb}" alt="Degenkolb Engineers"
                   style="height:38px;display:block;">
            </a>
          </div>
          <p style="text-align:center;color:#9a9a9a;font-size:0.72rem;
                    margin-top:0.85rem;">
            Built by Degenkolb Engineers on the NHERI SimCenter
            Damage and Loss Model Library.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
