"""
theme.py
--------
The visual design system for the DLML Explorer.

Streamlit only lets you set *one* default theme (in ``.streamlit/config.toml``),
so a true in-app light/dark switch has to be done as a CSS overlay driven by
session state. That is what this module does:

* ``apply_theme()`` injects the global stylesheet on every page run. It always
  defines the **light** palette (the default), and when the dark flag is set it
  appends a second layer that re-binds the same CSS custom properties to the
  **dark** palette. Because every rule is written against ``var(--…)`` tokens,
  flipping the palette restyles the whole app — no per-rule duplication.
* ``render_theme_toggle()`` is the manual switch (defaults to light) and lives
  in the sidebar so it is present on every page.

Accents differ by theme on purpose: a deep electric blue on light, a luminous
cyan glow on dark.

The hero/header structure lives in :mod:`st_ui.branding`; the matching
``.dlml-hero*`` rules live here so all styling stays in one place.
"""

from __future__ import annotations

import streamlit as st

# Session-state flag for the manual dark-mode toggle (default: light).
_DARK_KEY = "dlml_dark"

# Google Fonts — loaded from the viewer's browser (no CSP restriction here):
# Inter for UI text, JetBrains Mono for component IDs / code chips.
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700;800&"
    "family=JetBrains+Mono:wght@400;500;600&display=swap');"
)

# ── Palettes ──────────────────────────────────────────────────────────────────
# Light is the default; dark is the toggled overlay. Same token names in both so
# the static CSS below never needs to branch on theme.

_LIGHT = {
    "--bg": "#F4F6FB",
    "--bg-grad-a": "#EEF2FB",
    "--bg-grad-b": "#F7F8FC",
    "--surface": "#FFFFFF",
    "--surface-2": "#EEF2F8",
    "--surface-3": "#E6EBF3",
    "--text": "#171C26",
    "--text-muted": "#5B6573",
    "--border": "rgba(16,24,40,0.10)",
    "--border-strong": "rgba(16,24,40,0.18)",
    "--accent": "#E74646",
    "--accent-2": "#2B67E9",
    "--accent-contrast": "#FFFFFF",
    "--accent-soft": "rgba(37,99,235,0.10)",
    "--accent-glow": "rgba(37,99,235,0.18)",
    "--shadow-sm": "0 1px 2px rgba(16,24,40,0.06)",
    "--shadow": "0 1px 3px rgba(16,24,40,0.07), 0 12px 28px rgba(16,24,40,0.06)",
    "--hero-text": "#FFFFFF",
    "--code-text": "#1D4ED8",
}

_DARK = {
    "--bg": "#0A0D14",
    "--bg-grad-a": "#0E1422",
    "--bg-grad-b": "#0A0D14",
    "--surface": "#141A27",
    "--surface-2": "#1B2333",
    "--surface-3": "#232C40",
    "--text": "#E7F3EC",
    "--text-muted": "#97B6A3",
    "--border": "rgba(255,255,255,0.09)",
    "--border-strong": "rgba(255,255,255,0.16)",
    "--accent": "#810505",
    "--accent-2": "#002B7C",
    "--accent-contrast": "#04141B",
    "--accent-soft": "rgba(34,211,238,0.14)",
    "--accent-glow": "rgba(34,211,238,0.26)",
    "--shadow-sm": "0 1px 2px rgba(0,0,0,0.4)",
    "--shadow": "0 2px 6px rgba(0,0,0,0.45), 0 18px 40px rgba(0,0,0,0.40)",
    "--hero-text": "#04141B",
    "--code-text": "#7DE3F4",
    "--interact-button": "#A9C0FF",
}


def _vars_block(selector: str, variables: dict[str, str]) -> str:
    """Render a CSS rule binding the palette tokens on *selector*."""
    body = " ".join(f"{name}:{value};" for name, value in variables.items())
    return f"{selector}{{{body}}}"


# ── Static stylesheet ──────────────────────────────────────────────────────────
# Written entirely against the var(--…) tokens above, so it is palette-agnostic.
_STATIC_CSS = """
html, body, .stApp, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Canvas: a soft diagonal wash rather than a flat fill. */
.stApp {
  background:
    radial-gradient(1200px 600px at 100% -10%, var(--accent-soft), transparent 60%),
    linear-gradient(160deg, var(--bg-grad-a), var(--bg-grad-b));
  background-attachment: fixed;
  color: var(--text);
}

/* Translucent top bar so the canvas wash shows through. */
[data-testid="stHeader"] {
  background: transparent;
  backdrop-filter: blur(6px);
}
[data-testid="stToolbar"] { color: var(--text-muted); }

/* Roomier main column. */
.block-container,
[data-testid="stMainBlockContainer"] {
  padding-top: 2.4rem;
  padding-bottom: 4rem;
  max-width: 1180px;
}

/* Typography */
h1, h2, h3, h4 {
  color: var(--text);
  font-weight: 700;
  letter-spacing: -0.012em;
}
h2 { font-size: 1.55rem; margin-top: 0.4rem; }
h3 { font-size: 1.2rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
p, li, span, label, [data-testid="stMarkdownContainer"] { color: var(--text); }

/* Mono code chips — component IDs read like data, not prose. */
code, kbd {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
  background: var(--surface-2) !important;
  color: var(--code-text) !important;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.08em 0.4em;
  font-size: 0.86em;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
/* The sidebar is drag-resizable from its right edge, but Streamlit caps the
   expanded width at 600px. Raise that ceiling so it can be dragged to at least
   half the page — handy for viewing the Added Components panels (charts +
   parameter tables) side by side with the main content. A viewport-relative
   cap (not a fixed px) guarantees it always clears 50% on any monitor; scoped
   to the expanded state (aria-expanded="true") so the collapse control still
   works. */
[data-testid="stSidebar"][aria-expanded="true"] {
  max-width: 60vw !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
/* Descendant combinator (not ">"): a button with a `help` tooltip is wrapped
   in an extra div, so a direct-child selector would miss it — which is what
   left the download buttons unstyled (white) in dark mode. */
.stButton button,
.stDownloadButton button {
  border-radius: 10px;
  border: 1px solid var(--border-strong);
  background: var(--surface);
  color: var(--text);
  font-weight: 600;
  transition: transform .08s ease, box-shadow .15s ease,
              border-color .15s ease, background .15s ease;
}
.stButton button:hover,
.stDownloadButton button:hover {
  border-color: var(--accent);
  color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
  transform: translateY(-1px);
}
/* Primary / download actions get the accent fill. */
.stButton button[kind="primary"],
[data-testid="stBaseButton-primary"],
.stDownloadButton button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: var(--accent-contrast);
  border: none;
  box-shadow: 0 6px 16px var(--accent-glow);
}
.stButton button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
.stDownloadButton button:hover {
  filter: brightness(1.05);
  box-shadow: 0 8px 22px var(--accent-glow);
}
/* The download buttons use a flat solid fill (no gradient), matching the
   "Add component" button, with a white label that stays legible in both
   themes (the dark theme's --accent-contrast is a near-black ink that would
   be unreadable on the fill). */
.stDownloadButton button,
.stDownloadButton button:hover {
  background: var(--accent-2) !important;
  background-image: none !important;
  color: #ffffff !important;
}
.stButton button:disabled,
.stButton button:disabled:hover {
  opacity: 0.55; transform: none; box-shadow: none;
  border-color: var(--border); color: var(--text-muted);
}
/* The component-detail "Add component" button uses a flat solid fill instead of
   the accent gradient other primary actions get. Targeted via the
   st-key-…add_btn_… wrapper class Streamlit adds for the keyed widget. */
[class*="st-key-"][class*="add_btn_"] button {
  background: var(--accent-2) !important;
  background-image: none !important;
  color: #ffffff !important;
  box-shadow: var(--shadow-sm) !important;
}
/* Keep the filled-action button LABELS white in both themes. The label is an
   inner <p>/markdown element, and the global text rule colors it with --text
   directly — which an inherited color (even !important on the <button>) can't
   override. So target the descendants explicitly. In light mode --text is dark,
   which is what made these labels read dark on the accent fill. */
.stDownloadButton button, .stDownloadButton button *,
[class*="st-key-"][class*="add_btn_"] button,
[class*="st-key-"][class*="add_btn_"] button * {
  color: #ffffff !important;
}
[class*="st-key-"][class*="add_btn_"] button:hover {
  filter: brightness(1.06);
  box-shadow: var(--shadow-sm) !important;
}

/* ── Inputs & selects ────────────────────────────────────────────────────── */
[data-baseweb="input"], [data-baseweb="select"] > div, .stTextInput input,
[data-baseweb="textarea"], textarea {
  background: var(--surface) !important;
  border-radius: 10px !important;
  border: 1px solid var(--border-strong) !important;
  color: var(--text) !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
.stTextInput input::placeholder { color: var(--text-muted); }

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
}
/* The header summary and the expanded content body otherwise inherit
   Streamlit's secondaryBackgroundColor (white) — which the dark overlay can't
   reach — making expanders flash white when opened in dark mode. Force them
   transparent so they show the card's own --surface in both themes. */
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpanderDetails"] {
  background: transparent !important;
}
[data-testid="stExpander"] summary:hover { color: var(--accent); }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--border); }
[data-baseweb="tab"] { font-weight: 600; color: var(--text-muted); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
[data-baseweb="tab-highlight"] { background: var(--accent) !important; }

/* ── Radios as a segmented control ───────────────────────────────────────── */
div[role="radiogroup"] {
  gap: 0.4rem;
  background: var(--surface-2);
  padding: 0.3rem;
  border-radius: 12px;
  border: 1px solid var(--border);
  width: fit-content;
}
div[role="radiogroup"] label {
  border-radius: 9px;
  padding: 0.28rem 0.8rem;
  margin: 0 !important;
  transition: background .12s ease, color .12s ease;
}
div[role="radiogroup"] label:hover { background: var(--surface-3); }

/* ── Progress (relevance bars) ───────────────────────────────────────────── */
[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Plotly: frame charts on the theme surface (white in light, dark in dark).
   The figures themselves switch templates to match — see st_visuals/figures.py
   and the dark flag threaded from st_core/component.py. ──────────────────── */
[data-testid="stPlotlyChart"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.5rem;
  box-shadow: var(--shadow-sm);
}

/* ── Dataframes ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border-radius: 12px;
  border: 1px solid var(--border);
  overflow: hidden;
}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
*::-webkit-scrollbar { width: 11px; height: 11px; }
*::-webkit-scrollbar-thumb {
  background: var(--surface-3);
  border-radius: 8px;
  border: 2px solid transparent;
  background-clip: content-box;
}
*::-webkit-scrollbar-thumb:hover { background: var(--accent); background-clip: content-box; }
*::-webkit-scrollbar-track { background: transparent; }

/* Trim Streamlit chrome. */
[data-testid="stStatusWidget"] { display: none; }
footer { display: none; }

/* ════════════════════════════════════════════════════════════════════════
   Hero / header (structure rendered by st_ui.branding.render_header)
   ════════════════════════════════════════════════════════════════════════ */
.dlml-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
  flex-wrap: wrap;
  padding: 1.9rem 2.1rem;
  margin: 0.2rem 0 1.6rem;
  border-radius: 20px;
  background:
    radial-gradient(900px 300px at 110% -40%, rgba(255,255,255,0.18), transparent 70%),
    linear-gradient(125deg, var(--accent), var(--accent-2));
  box-shadow: 0 16px 40px var(--accent-glow);
  position: relative;
  overflow: hidden;
}
.dlml-hero::after {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(var(--hero-text) 1px, transparent 1px),
    linear-gradient(90deg, var(--hero-text) 1px, transparent 1px);
  background-size: 26px 26px;
  opacity: 0.05;
  pointer-events: none;
}
.dlml-hero-main { position: relative; z-index: 1; }
.dlml-hero-eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--hero-text);
  opacity: 0.82;
  margin-bottom: 0.35rem;
}
.dlml-hero-title {
  font-size: 2.7rem;
  font-weight: 800;
  line-height: 1.02;
  letter-spacing: -0.03em;
  color: var(--hero-text);
  margin: 0;
}
.dlml-hero-subtitle {
  font-size: 1.06rem;
  font-weight: 500;
  color: var(--hero-text);
  opacity: 0.9;
  margin: 0.45rem 0 0;
}
/* Logo card stays white in both themes so the dark Degenkolb wordmark reads. */
.dlml-logo-card {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  gap: 1.4rem;
  background: #ffffff;
  border-radius: 12px;
  padding: 0.55rem 1.15rem;
  box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

@media (max-width: 640px) {
  .dlml-hero-title { font-size: 2.05rem; }
  .dlml-hero { padding: 1.4rem 1.4rem; }
}
"""


def _dark_active() -> bool:
    """True when the manual dark-mode toggle is on (default: light)."""
    return bool(st.session_state.get(_DARK_KEY, False))


def is_dark() -> bool:
    """
    Public: whether the dark theme is active for this run.

    Used by chart code (``st_visuals.figures``, via ``st_core.component``) to
    pick a matching Plotly template, since our dark mode is a CSS overlay that
    Streamlit's native chart theming can't see.
    """
    return _dark_active()


def apply_theme() -> None:
    """
    Inject the global stylesheet for this page run.

    Call this once at the top of every page (right after ``set_page_config``).
    Light tokens are always defined; the dark overlay is appended only when the
    toggle is on, so the same rules restyle the whole app either way.
    """
    css = _FONT_IMPORT + _vars_block(":root", _LIGHT)
    if _dark_active():
        # Higher specificity + later source order → dark wins over the :root light tokens.
        css += _vars_block(".stApp", _DARK)
    css += _STATIC_CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def use_full_width() -> None:
    """
    Stretch the main content column to the full page width.

    The base stylesheet caps content at a readable ~1180px, which suits
    prose-heavy pages like About. Data-dense pages — the search panel, browse
    tree, parameter tables, and charts — should use the extra space on large
    monitors, so the home page calls this to lift the cap (keeping comfortable
    side gutters). Call it after :func:`apply_theme`; the ``!important`` rule
    plus later source order win over the base cap.
    """
    st.markdown(
        """
        <style>
        .block-container,
        [data-testid="stMainBlockContainer"] {
          max-width: 100% !important;
          padding-left: clamp(1rem, 3vw, 3.5rem) !important;
          padding-right: clamp(1rem, 3vw, 3.5rem) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_theme_toggle() -> None:
    """
    Render the manual light/dark switch.

    Mounted in the sidebar so it appears on every page. The widget writes
    ``st.session_state[_DARK_KEY]`` before the next rerun, which
    :func:`apply_theme` reads at the top of that run.
    """
    st.toggle(
        "🌙 Dark mode",
        key=_DARK_KEY,
    )
