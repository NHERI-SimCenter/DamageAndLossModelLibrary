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


def render_header(title: str, subtitle: str | None = None) -> None:
    """
    Render the page hero — an editorial header: the title with a short
    SimCenter-red rule, an optional subtitle, and a one-line note making it
    clear the Explorer is managed by the SimCenter and was built with
    Degenkolb's support. The contributor logos live in the sticky top bar
    (:func:`render_top_bar`) and the footer, so the hero stays text-only.

    Emitted as one HTML block so the typography and red rule style as a unit;
    the matching ``.dlml-hero*`` CSS lives in :mod:`dlml.web.st_ui.theme`.
    """
    # Flush-left HTML (no leading whitespace on any line) — st.markdown parses
    # the string as Markdown first, and any line indented 4+ spaces becomes a
    # literal code block instead of rendered HTML. The contributor logos live in
    # the sticky top bar (render_top_bar) and the footer, not here.
    subtitle_html = (
        f'<p class="dlml-hero-subtitle">{subtitle}</p>' if subtitle else ""
    )
    st.markdown(
        '<div class="dlml-hero">'
        '<div class="dlml-hero-main">'
        f'<h1 class="dlml-hero-title">{title}</h1>'
        f'{subtitle_html}'
        '<p class="dlml-hero-brandline">Managed by the NHERI SimCenter'
        ' · developed with support from Degenkolb Engineers</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_top_bar() -> None:
    """
    Render the slim, sticky brand bar shown on every page.

    Left: the two contributor logos on a white card and the "DLML Explorer"
    wordmark. Right: the page links and the light/dark switch. The
    ``.st-key-dlml-topbar`` rules (see :mod:`dlml.web.st_ui.theme`) pin it to the
    top so the branding and nav stay visible on scroll, and Streamlit's own
    header chrome is hidden. The sidebar is left entirely for selected models.
    Page-link paths are relative to the entry script (``app.py``, the About
    landing page).
    """
    from dlml.web.st_ui.theme import render_theme_toggle

    # Brand block (flush-left HTML — see render_header): the two contributor
    # logos on a white card, then the "DLML Explorer" wordmark.
    logos = _logos()
    if logos is not None:
        simcenter, degenkolb = logos
        logos_html = (
            '<span class="dlml-brand-logos">'
            f'<a href="{_SIMCENTER_URL}" target="_blank" rel="noopener" '
            'title="NSF NHERI SimCenter">'
            f'<img src="{simcenter}" alt="NSF NHERI SimCenter" '
            'style="height:22px;display:block;"></a>'
            f'<a href="{_DEGENKOLB_URL}" target="_blank" rel="noopener" '
            'title="Degenkolb Engineers">'
            f'<img src="{degenkolb}" alt="Degenkolb Engineers" '
            'style="height:16px;display:block;"></a>'
            '</span>'
        )
    else:
        logos_html = ""
    brand_html = (
        '<div class="dlml-brand">'
        + logos_html
        + '<span class="dlml-brand-name">DLML Explorer</span>'
        + '</div>'
    )

    # Keyed container so the theme can pin this bar to the top and keep its
    # columns from wrapping (see the .st-key-dlml-topbar rules). The empty
    # marker div is what the sticky CSS selector keys off (the proven Streamlit
    # sticky-header recipe: target the flow element that *contains* the marker).
    # Order: brand, Browse, About, toggle — left-packed. The theme makes the row
    # wrap (not full-stack) and pushes the toggle to the right, so on a narrow
    # bar the toggle wraps first, then the two links join it on the second row.
    with st.container(key="dlml-topbar"):
        st.markdown('<div class="dlml-sticky-marker"></div>', unsafe_allow_html=True)
        brand, browse, about, toggle = st.columns(
            [5, 2.6, 1.3, 2.2], gap="small", vertical_alignment="center"
        )
        with brand:
            st.markdown(brand_html, unsafe_allow_html=True)
        browse.page_link(
            "pages/1_Browse_and_Search.py", label="Browse & Search", icon="🔍"
        )
        about.page_link("app.py", label="About", icon="ℹ️")
        with toggle:
            render_theme_toggle()


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
                       text-transform:uppercase;"></span>
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
                    margin-bottom:0.9rem;"></p>
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
            Built with the support of Degenkolb Engineers on the NHERI SimCenter
            Damage and Loss Model Library.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
