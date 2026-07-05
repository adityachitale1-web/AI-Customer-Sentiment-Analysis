"""CXSentinel branding: name, logo, Liquid Ether background and global CSS."""

APP_NAME = "CXSentinel"
TAGLINE = "Never Miss a Customer Signal."

# Sentinel shield with a sentiment pulse — Liquid Ether palette
LOGO_SVG = """
<svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="cxs-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#5227FF"/>
      <stop offset="55%" stop-color="#B19EEF"/>
      <stop offset="100%" stop-color="#FF9FFC"/>
    </linearGradient>
  </defs>
  <path d="M32 5 L55 13 V30 C55 45 45.5 55 32 59.5 C18.5 55 9 45 9 30 V13 Z"
        stroke="url(#cxs-grad)" stroke-width="4.5" stroke-linejoin="round"
        fill="rgba(82,39,255,0.10)"/>
  <path d="M17 33 H23 L27 25 L32 42 L37 28 L40 33 H47"
        stroke="url(#cxs-grad)" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def logo_html(size: int = 40) -> str:
    # Single-line HTML: st.markdown treats indented multi-line HTML as a
    # Markdown code block, so all whitespace is collapsed.
    return " ".join(LOGO_SVG.format(size=size).split())


def wordmark(size: int = 34) -> str:
    span_style = (
        f"font-size:{size * 0.72:.0f}px;font-weight:800;letter-spacing:-.02em;"
        "background:linear-gradient(90deg,#F2EFFD 30%,#B19EEF 70%,#FF9FFC 100%);"
        "-webkit-background-clip:text;background-clip:text;color:transparent;"
    )
    return ('<div style="display:flex;align-items:center;gap:.6rem;">'
            f'{logo_html(size)}<span style="{span_style}">{APP_NAME}</span></div>')


# The real ReactBits LiquidEther fluid simulation (ported to vanilla JS)
# lives in liquid_ether.html next to this file. Read per call so edits go
# live on the next page load without a server restart.
from pathlib import Path as _Path


def liquid_ether_html() -> str:
    return (_Path(__file__).resolve().parent / "liquid_ether.html").read_text()

GLOBAL_CSS = """
<style>
.stApp { background: #000000; }
/* Pin the WebGL iframe as a fixed, non-interactive backdrop */
iframe[title="st.components.v1.html"], iframe[title="st.iframe"] {
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    z-index: 0; border: none; pointer-events: none;
    /* Darken the fluid so foreground text stays readable */
    filter: brightness(0.5);
}
div[data-testid="stElementContainer"]:has(> iframe[title="st.components.v1.html"]),
div[data-testid="stElementContainer"]:has(> iframe[title="st.iframe"]),
div[data-testid="element-container"]:has(> iframe[title="st.iframe"]) {
    height: 0; min-height: 0;
}
/* Lift the app content above the backdrop; a dark veil keeps text readable
   everywhere, even where no card sits behind it */
[data-testid="stAppViewContainer"] {
    position: relative; z-index: 1;
    background: rgba(0, 0, 0, 0.50);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(11, 5, 28, 0.95); }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(8, 3, 22, 0.82);
    border-radius: 12px;
}
/* Top navigation tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: .4rem; background: rgba(8,3,22,.82);
    border: 1px solid rgba(177,158,239,.22); border-radius: 999px;
    padding: .3rem .5rem; width: fit-content;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px; padding: .35rem 1.2rem; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #5227FF 0%, #7A4DFF 100%) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
/* Buttons: gradient pill for primary */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #5227FF 0%, #7A4DFF 100%);
    border: none; border-radius: 999px; font-weight: 700;
    padding: .55rem 1.6rem;
    box-shadow: 0 4px 24px rgba(82, 39, 255, .45);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #6A3DFF 0%, #9A6DFF 100%);
    box-shadow: 0 4px 32px rgba(122, 77, 255, .65);
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(177,158,239,.45);
    border-radius: 999px; font-weight: 600;
}
/* Official Google "G" mark on the Continue With Google button */
[class*="st-key-google_btn"] button p::before {
    content: ""; display: inline-block; width: 1.1em; height: 1.1em;
    margin-right: .6rem; vertical-align: -0.18em;
    background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="%23EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/><path fill="%234285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="%23FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/><path fill="%2334A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/></svg>') no-repeat center / contain;
}
[class*="st-key-google_btn"] button {
    background: #FFFFFF !important; color: #1F1F1F !important;
    border: 1px solid #DADCE0 !important; border-radius: 999px !important;
    font-weight: 600 !important;
}
/* Landing building blocks */
.sl-hero { text-align: center; padding: 2.6rem 1rem 1rem; }
.sl-hero h1 {
    font-size: clamp(2.4rem, 6vw, 4.2rem); font-weight: 800; line-height: 1.08;
    letter-spacing: -.02em; margin: 0 0 1rem;
}
.sl-grad-text {
    background: linear-gradient(90deg, #F2EFFD 20%, #B19EEF 60%, #FF9FFC 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.sl-hero p { font-size: 1.15rem; color: #E4DEF8; max-width: 620px; margin: 0 auto 1.6rem;
             text-shadow: 0 1px 10px rgba(6,0,16,.9); }
.sl-badge {
    display: inline-block; padding: .35rem 1rem; border-radius: 999px;
    border: 1px solid rgba(177,158,239,.4); background: rgba(82,39,255,.15);
    color: #D8CFF7; font-size: .85rem; font-weight: 600; margin-bottom: 1.3rem;
}
.sl-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem; margin: 2.2rem 0;
}
.sl-stat {
    background: rgba(8,3,22,.85);
    border: 1px solid rgba(177,158,239,.22); border-radius: 16px;
    padding: 1.2rem 1rem; text-align: center;
}
.sl-stat .v { font-size: 1.9rem; font-weight: 800; color: #F2EFFD; }
.sl-stat .l { font-size: .82rem; color: #CFC6EC; margin-top: .25rem; }
.sl-cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 1rem; margin: 1rem 0 2rem;
}
.sl-card {
    background: rgba(8,3,22,.85);
    border: 1px solid rgba(177,158,239,.22); border-radius: 16px;
    padding: 1.4rem 1.3rem; transition: transform .15s, border-color .15s;
}
.sl-card:hover { transform: translateY(-3px); border-color: rgba(255,159,252,.55); }
.sl-card .ic { font-size: 1.7rem; }
.sl-card h4 { margin: .55rem 0 .35rem; color: #F2EFFD; font-size: 1.05rem; }
.sl-card p { margin: 0; color: #D7CFEF; font-size: .9rem; line-height: 1.45; }
.sl-step {
    display: flex; gap: .9rem; align-items: flex-start; margin-bottom: 1rem;
    background: rgba(8,3,22,.85);
    border: 1px solid rgba(177,158,239,.18); border-radius: 14px;
    padding: .95rem 1.1rem;
}
.sl-step .n {
    min-width: 2rem; height: 2rem; border-radius: 999px; display: flex;
    align-items: center; justify-content: center; font-weight: 800; color: white;
    background: linear-gradient(135deg, #5227FF, #FF9FFC);
}
.sl-step h5 { margin: 0 0 .15rem; color: #F2EFFD; font-size: 1rem; }
.sl-step p { margin: 0; color: #D7CFEF; font-size: .88rem; }
.sl-footer {
    margin-top: 2.5rem; padding: 1.6rem 0 .8rem; text-align: center;
    border-top: 1px solid rgba(177,158,239,.18); color: #C3B8E4; font-size: .85rem;
    text-shadow: 0 1px 8px rgba(6,0,16,.9);
}
h2.sl-section { text-align: center; font-weight: 800; letter-spacing: -.01em; margin: 1.2rem 0 .4rem;
                text-shadow: 0 1px 12px rgba(6,0,16,.9); }
p.sl-section-sub { text-align: center; color: #D3CAEE; margin: 0 0 1.4rem;
                   text-shadow: 0 1px 8px rgba(6,0,16,.9); }
.sl-divider { display:flex; align-items:center; gap:.8rem; color:#B7ACD9; font-size:.8rem; margin:.6rem 0; }
.sl-divider::before, .sl-divider::after { content:""; flex:1; height:1px; background:rgba(177,158,239,.25); }
</style>
"""
