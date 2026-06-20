"""
Shared visual styling for the AuNP synthesis tool (dark theme).

Call ``apply_base_style()`` once per page (right after ``st.set_page_config``). The look is
deliberately restrained and academic: IBM Plex typefaces on a deep slate background, a muted
gold accent (fitting the gold-nanoparticle subject), soft cards, and instrument-style numeric
readouts — no emojis, no animation, no marketing gloss.
"""

import streamlit as st

# Palette (dark)
GOLD = "#C9A24C"    # accent (section rules, links, highlights)
TEXT = "#EAECEF"    # primary text / headings
MUTED = "#9AA3AF"   # secondary text
CARD_BORDER = "#2A323D"

_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, p, li, label, input, button, select, textarea {
    font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
}
h1, h2, h3, h4, h5 {
    font-family: 'IBM Plex Serif', Georgia, serif;
    letter-spacing: -0.01em;
    color: #EAECEF;
}

/* Keep the hamburger minimal via config; only drop the footer here. Crucially, do NOT hide
   the toolbar/header region — that also hides the control that re-opens a collapsed sidebar. */
footer { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] { visibility: visible !important; opacity: 1 !important; }

.block-container { padding-top: 2.4rem; }

/* numeric readouts: instrument feel */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    color: #EAECEF;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9AA3AF;
}

/* bordered containers -> soft cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px;
    border-color: #2A323D !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.30);
}

/* buttons: quiet, professional */
div.stButton > button, div.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.15rem;
    transition: background-color .15s ease, border-color .15s ease, color .15s ease;
}

a { color: #D8B768 !important; text-decoration: none; }
a:hover { text-decoration: underline; }

/* page header block */
.ah-kicker { text-transform: uppercase; letter-spacing: .14em; font-size: .72rem;
             color: #C9A24C; font-weight: 600; margin-bottom: .35rem; }
.ah-title  { font-family:'IBM Plex Serif',serif; font-size: 2.05rem; font-weight: 600;
             color:#EFF1F4; margin: 0 0 .4rem 0; line-height: 1.16; }
.ah-sub    { color:#9AA3AF; font-size: 1.02rem; max-width: 66ch; margin: 0; line-height: 1.5; }
.ah-rule   { height: 3px; width: 56px; background:#C9A24C; border-radius: 2px; margin: 1rem 0 .3rem; }

/* section title */
.sec-wrap  { display:flex; align-items:baseline; gap:.6rem; margin: .4rem 0 .7rem;
             border-left: 3px solid #C9A24C; padding-left: .75rem; }
.sec-title { font-family:'IBM Plex Serif',serif; font-size: 1.28rem; font-weight: 600; color:#EAECEF; }
.sec-cap   { font-size: .85rem; color:#8A93A0; }

/* status pill */
.pill { display:inline-block; font-weight:600; font-size:.92rem; padding:.3rem .8rem;
        border-radius: 999px; }

/* small label above a value */
.field-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:#9AA3AF; margin-bottom:.2rem; }
</style>
"""


def apply_base_style():
    """Inject the shared stylesheet. Call once, right after st.set_page_config."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


def page_header(title, subtitle=None, kicker=None):
    """Render the page's main header block (kicker / title / subtitle / accent rule)."""
    parts = ['<div class="ah-head">']
    if kicker:
        parts.append(f'<div class="ah-kicker">{kicker}</div>')
    parts.append(f'<div class="ah-title">{title}</div>')
    if subtitle:
        parts.append(f'<p class="ah-sub">{subtitle}</p>')
    parts.append('<div class="ah-rule"></div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def section_title(text, caption=None):
    """Render a left-ruled section heading with an optional muted caption."""
    cap = f'<span class="sec-cap">{caption}</span>' if caption else ""
    st.markdown(f'<div class="sec-wrap"><span class="sec-title">{text}</span>{cap}</div>',
                unsafe_allow_html=True)


def status_pill(success):
    """Return HTML for a coloured Successful/Unsuccessful pill (no icons/emojis)."""
    if success:
        return ('<span class="pill" style="background:#15271D;color:#7FD3A6;'
                'border:1px solid rgba(127,211,166,.30);">Successful</span>')
    return ('<span class="pill" style="background:#2C1C18;color:#E89E86;'
            'border:1px solid rgba(232,158,134,.30);">Unsuccessful</span>')
