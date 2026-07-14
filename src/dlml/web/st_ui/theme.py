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

The hero/header structure lives in :mod:`dlml.web.st_ui.branding`; the matching
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

# Brand colors, sampled from the NSF NHERI SimCenter logo: the SimCenter red
# (#C84038) is the signature accent used throughout; the NHERI-triangle blue is
# the cool secondary reserved for filled utility actions (Add / Download). The
# -soft / -glow tints are derived from the red so focus rings and washes read as
# one family.
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
    "--accent": "#C8382E",
    "--accent-2": "#1A5CC4",
    "--accent-contrast": "#FFFFFF",
    "--accent-soft": "rgba(200,56,46,0.10)",
    "--accent-glow": "rgba(200,56,46,0.20)",
    "--shadow-sm": "0 1px 2px rgba(16,24,40,0.06)",
    "--shadow": "0 1px 3px rgba(16,24,40,0.07), 0 12px 28px rgba(16,24,40,0.06)",
    "--code-text": "#1A5CC4",
}

_DARK = {
    "--bg": "#0A0D14",
    "--bg-grad-a": "#0E1422",
    "--bg-grad-b": "#0A0D14",
    "--surface": "#141A27",
    "--surface-2": "#1B2333",
    "--surface-3": "#232C40",
    "--text": "#E8ECF3",
    "--text-muted": "#9BA6B8",
    "--border": "rgba(255,255,255,0.09)",
    "--border-strong": "rgba(255,255,255,0.16)",
    "--accent": "#E15A4B",
    "--accent-2": "#5A8DE8",
    "--accent-contrast": "#FFFFFF",
    "--accent-soft": "rgba(225,90,75,0.16)",
    "--accent-glow": "rgba(225,90,75,0.28)",
    "--shadow-sm": "0 1px 2px rgba(0,0,0,0.4)",
    "--shadow": "0 2px 6px rgba(0,0,0,0.45), 0 18px 40px rgba(0,0,0,0.40)",
    "--code-text": "#8FB4F2",
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

/* Main column: enough top padding for the brand bar (the first element) to
   clear Streamlit's translucent header; the bar then sticks at that same offset
   (see the .st-key-dlml-topbar rule) so it pins just under the header. */
.block-container,
[data-testid="stMainBlockContainer"] {
  padding-top: 2.9rem;
  padding-bottom: 2.2rem;
  max-width: 1180px;
}

/* ── Vertical rhythm ─────────────────────────────────────────────────────────
   Streamlit stacks every element in a flex column with a uniform ~1rem gap,
   which reads as a lot of ambient whitespace on data-dense pages. Tighten the
   default so elements that belong together sit close; genuine section breaks
   are carried deliberately by dividers, card borders, and the hero margin —
   not by ambient space. !important because the gap ships on an emotion class we
   need to beat regardless of its specificity. */
[data-testid="stVerticalBlock"] { gap: 0.6rem !important; }
/* Trim only the LAST paragraph's trailing margin so inter-element spacing is
   governed by the gap above rather than doubled by it. Multi-paragraph prose
   keeps its internal paragraph spacing untouched. */
[data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }

/* Section labels ("Model metadata", "Damage states", …) read as headers, so
   they get a little more air than the base gap — above (to break from the
   previous group) and below (to sit apart from their own content). Padding, not
   margin, so it can't collapse away. The model description that follows the
   metadata label gets matching breathing room before the Technical-notes row. */
.dlml-section-label { font-weight: 700; padding: 0.35rem 0 0.3rem; }
.dlml-model-desc    { padding-bottom: 0.4rem; }

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
/* Navigation lives in the custom top bar (see st_ui.branding.render_top_bar),
   so hide Streamlit's automatic multipage list — the sidebar is reserved for
   the user's selected models. */
[data-testid="stSidebarNav"] { display: none; }
/* The sidebar collapse control ("«") is a Material *font* glyph, so it takes
   `color`, not `fill`; the default dark ink vanishes against the dark sidebar.
   Target the button and all its descendants (the icon span) so it stays visible
   in both modes. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] * {
  color: var(--text) !important;
  fill: var(--text) !important;
  opacity: 0.9;
}

/* Top bar columns keep their content width (no forced shrink) so the nav labels
   never clip; on a genuinely narrow window they wrap to a second row instead of
   being cut off. */
/* position:sticky only engages if no ancestor between the sticky element and
   the scroll container clips it. Streamlit puts the content in nested blocks;
   force the ones above the bar to overflow:visible so the sticky context
   reaches the scrolling main area. */
[data-testid="stMainBlockContainer"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
  overflow: visible !important;
}
/* Pin the brand bar just under Streamlit's header (top offset ≈ its height) so
   the logos, the "DLML Explorer" wordmark, and the nav stay visible on scroll.
   Two selectors for robustness: the keyed container class, and the proven
   marker recipe — the direct child of a vertical block that contains our
   sticky marker (that's the flow element Streamlit sticky needs). */
[class*="st-key-dlml-topbar"],
[data-testid="stVerticalBlock"] > div:has(.dlml-sticky-marker) {
  position: sticky;
  position: -webkit-sticky;
  top: 2.875rem;
  z-index: 100;
}
[class*="st-key-dlml-topbar"] {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  padding: 0.85rem 0.9rem;
  margin-bottom: 0.6rem;
}
.dlml-sticky-marker { height: 0; margin: 0; }
/* Fully collapse the marker's element container so it adds no height (its
   residual line-height was pushing the bar's contents toward the bottom). It
   stays in the DOM, so the :has() sticky selector still matches. */
[class*="st-key-dlml-topbar"] [data-testid="stElementContainer"]:has(.dlml-sticky-marker) {
  display: none !important;
}
[class*="st-key-dlml-topbar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
/* Center the logos, wordmark, and nav vertically; let the row WRAP (to at most
   two rows) rather than full-stack. Each column is content-width (no shrink →
   labels never clip), and the toggle is pushed to the right so it's the first
   to wrap, with the two links following onto the second row as the bar narrows. */
[class*="st-key-dlml-topbar"] [data-testid="stHorizontalBlock"] {
  align-items: center !important;
  flex-wrap: wrap !important;
  row-gap: 0.75rem;
}
[class*="st-key-dlml-topbar"] [data-testid="stColumn"] {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
}
[class*="st-key-dlml-topbar"] [data-testid="stColumn"]:last-child { margin-left: auto; }
/* Keep the brand block a fixed size — don't let the logos or wordmark shrink as
   the column reflows. */
.dlml-brand {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  flex-shrink: 0;
}
.dlml-brand-name {
  font-weight: 800;
  font-size: 1.08rem;
  letter-spacing: -0.015em;
  color: var(--text);
  white-space: nowrap;
}
/* White fill kept so the dark Degenkolb/SimCenter wordmarks read in dark mode,
   but no border — in light mode the fill matches the white bar, so the logos
   sit cleanly on the bar with no visible outline. */
.dlml-brand-logos {
  display: inline-flex;
  align-items: center;
  gap: 0.9rem;
  background: #ffffff;
  border-radius: 8px;
  padding: 0.3rem 0.6rem;
}
.dlml-brand-logos img { max-width: none !important; flex-shrink: 0; }
/* The sidebar is drag-resizable from its right edge, but Streamlit caps the
   expanded width at 600px. Raise that ceiling so it can be dragged to at least
   half the page — handy for viewing the selected-model panels (charts +
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
/* The About-page call-to-action into Browse & Search: a flat, filled button in
   the SimCenter red (the featured brand action), no gradient — easy to spot but
   calmer than the primary gradient. Utility buttons (Add / Download) stay blue. */
[class*="st-key-cta_browse"] button {
  background: var(--accent) !important;
  background-image: none !important;
  color: #ffffff !important;
  border: none !important;
  box-shadow: var(--shadow-sm) !important;
}
[class*="st-key-cta_browse"] button * { color: #ffffff !important; }
[class*="st-key-cta_browse"] button:hover {
  filter: brightness(1.05);
  box-shadow: 0 6px 16px var(--accent-glow) !important;
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
  margin-top: 0.25rem;
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
/* Trim the generous default padding so collapsed rows are compact and the body
   doesn't wrap a single caption in air. The reduced left inset also pulls the
   browse tree's nested cards closer together — tighter indentation reclaims
   horizontal space, and each card keeps its border as the nesting cue. */
[data-testid="stExpander"] summary { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
[data-testid="stExpanderDetails"] { padding: 0.5rem 0.4rem 0.8rem !important; }
/* Extra room above the About page's "Run it locally" expander, scoped to its
   keyed container so the browse tree stays tight. */
[class*="st-key-ril-block"] { margin-top: 0.7rem; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--border); }
[data-baseweb="tab"] { font-weight: 600; color: var(--text-muted); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
[data-baseweb="tab-highlight"] { background: var(--accent) !important; }
/* Pull the tab body up toward the tab strip instead of leaving a wide gap. */
[data-baseweb="tab-panel"] { padding-top: 0.4rem !important; }

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
/* A section break should read as intentional but compact — a hairline plus a
   little air, not a wide gutter. */
hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }

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

/* ── Parameter tables (themed HTML) ──────────────────────────────────────────
   The small fragility/consequence parameter grids render as plain HTML tables
   (see st_core.component._render_param_table), NOT st.dataframe, so headers
   size to fit their text and the table follows the theme — including dark mode,
   which the canvas-based st.dataframe cannot. Specificity + !important beat
   Streamlit's own markdown-table styling. */
.dlml-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow-sm);
  margin: 0.1rem 0 1rem;   /* breathing room below the table before the next section */
}
.dlml-table-wrap table.dlml-table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.84rem;
  margin: 0;
}
.dlml-table-wrap .dlml-table th,
.dlml-table-wrap .dlml-table td {
  padding: 0.32rem 0.7rem;
  white-space: nowrap;
  border-bottom: 1px solid var(--border);
  text-align: left;
  color: var(--text) !important;
  background: transparent !important;
}
.dlml-table-wrap .dlml-table thead th { background: var(--surface-2) !important; font-weight: 600; }
.dlml-table-wrap .dlml-table tbody th { color: var(--text-muted) !important; font-weight: 600; }
.dlml-table-wrap .dlml-table tbody tr:last-child th,
.dlml-table-wrap .dlml-table tbody tr:last-child td { border-bottom: none; }

/* ── Help tooltips ───────────────────────────────────────────────────────────
   The default hover box clips longer help text vertically; let it grow to fit
   and wrap at a comfortable width. */
[data-testid="stTooltipContent"] {
  max-height: none !important;
  max-width: 22rem !important;
  white-space: normal !important;
  overflow: visible !important;
  padding: 0.6rem 0.85rem 0.7rem !important;   /* the last line was flush to the bottom edge */
  line-height: 1.45 !important;
}

/* ── Responsive columns ──────────────────────────────────────────────────────
   Streamlit only collapses side-by-side columns to a single stack at a very
   narrow width — by which point the model panel's chart column has been
   squeezed to half a figure. Stack top-level columns sooner so wide content
   (charts, tables) drops to full width while it's still legible. The
   :not(…) excludes NESTED columns — e.g. the label/value metadata rows — which
   should stay side by side. Tune the breakpoint to taste. */
/* Make the main content area AND the sidebar query containers, so column
   stacking below reacts to the CONTENT width — which shrinks when the sidebar
   is widened, and is what actually constrains the sidebar's own panels —
   instead of the full viewport width (which doesn't). */
[data-testid="stMainBlockContainer"],
[data-testid="stSidebarUserContent"] { container-type: inline-size; }
@container (max-width: 900px) {
  /* Top-level columns stack when the available width is narrow — whether from a
     small window, a widened sidebar, or being inside the sidebar itself. Nested
     columns (label/value rows) are left alone, and the brand bar is excluded
     (it wraps to two rows instead — see the .st-key-dlml-topbar rules). */
  [data-testid="stColumn"]:not([data-testid="stColumn"] [data-testid="stColumn"]):not([class*="st-key-dlml-topbar"] *) {
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }
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

/* Trim Streamlit chrome. Keep stHeader intact — the sidebar collapse/expand
   controls live there, and hiding any of it can make a collapsed sidebar
   unrecoverable. Only the always-safe bits are removed. */
[data-testid="stStatusWidget"] { display: none; }
footer { display: none; }

/* ════════════════════════════════════════════════════════════════════════
   Hero / header (structure rendered by dlml.web.st_ui.branding.render_header)
   Editorial, text-only: the title sits on the page with a short SimCenter-red
   rule, the subtitle, and a one-line managed-by / built-with note. The logos
   live in the sticky top bar and the footer.
   ════════════════════════════════════════════════════════════════════════ */
.dlml-hero {
  padding: 0.2rem 0 1.1rem;
  margin: 0.1rem 0 1.3rem;
  border-bottom: 1px solid var(--border);
}
.dlml-hero-main { min-width: 0; }
.dlml-hero-title {
  font-size: 2.6rem;
  font-weight: 800;
  line-height: 1.04;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 0;
}
/* Short red rule under the title — the strategic brand accent. */
.dlml-hero-title::after {
  content: "";
  display: block;
  width: 54px;
  height: 3px;
  margin-top: 0.7rem;
  border-radius: 2px;
  background: var(--accent);
}
.dlml-hero-subtitle {
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--text-muted);
  margin: 0.85rem 0 0;
}
.dlml-hero-brandline {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin: 0.7rem 0 0;
}

@media (max-width: 640px) {
  .dlml-hero-title { font-size: 2.05rem; }
}
"""


# ── Dark-only overrides ─────────────────────────────────────────────────────
# A few widgets ship a fixed light look that our var-based overlay can't reach:
# st.code carries a light syntax-highlighter background, and the cache spinner a
# white pill. Recolor them ONLY in dark mode (so light mode keeps its native
# styling). Highlighter token colors are forced to the theme text — the trivial
# shell snippets don't need per-token color, and it keeps them legible on dark.
_DARK_ONLY_CSS = """
[data-testid="stCode"] { background: var(--surface-2) !important; }
[data-testid="stCode"] pre,
[data-testid="stCode"] code,
[data-testid="stCode"] span {
  background: transparent !important;
  color: var(--text) !important;
}
[data-testid="stSpinner"],
[data-testid="stSpinner"] > div {
  background: transparent !important;
  color: var(--text) !important;
}
"""


def _dark_active() -> bool:
    """True when the manual dark-mode toggle is on (default: light)."""
    return bool(st.session_state.get(_DARK_KEY, False))


def is_dark() -> bool:
    """
    Public: whether the dark theme is active for this run.

    Used by chart code (``dlml.web.st_visuals.figures``, via
    ``dlml.web.st_core.component``) to pick a matching Plotly template, since
    our dark mode is a CSS overlay that Streamlit's native chart theming can't
    see.
    """
    return _dark_active()


def apply_theme() -> None:
    """
    Inject the global stylesheet for this page run.

    Call this once at the top of every page (right after ``set_page_config``).
    Light tokens are always defined; the dark overlay is appended only when the
    toggle is on, so the same rules restyle the whole app either way.
    """
    # Persist the dark-mode choice across page switches. Streamlit drops widget
    # state for a key that hasn't been re-instantiated yet on the newly-opened
    # page; the toggle widget isn't built until later in this run, so touching
    # the key here keeps the choice from resetting to light when navigating
    # between Browse & Search and About.
    if _DARK_KEY in st.session_state:
        st.session_state[_DARK_KEY] = st.session_state[_DARK_KEY]

    css = _FONT_IMPORT + _vars_block(":root", _LIGHT)
    if _dark_active():
        # Higher specificity + later source order → dark wins over the :root light tokens.
        css += _vars_block(".stApp", _DARK)
    css += _STATIC_CSS
    if _dark_active():
        css += _DARK_ONLY_CSS
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
