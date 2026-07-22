"""
TechTrinetra7 — Brand Theme
Navy / Cyan / Violet dark theme used across all TechTrinetra7 deliverables.
"""

NAVY = "#0A0E1A"
NAVY_LIGHT = "#121829"
CARD_BG = "#161D30"
CYAN = "#00D4FF"
VIOLET = "#7C3AED"
TEXT = "#E6EAF2"
MUTED = "#8A93A6"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"

STAGE_COLORS = {
    "ingestion": CYAN,
    "chunking": "#38BDF8",
    "embedding": VIOLET,
    "dense": "#22C55E",
    "sparse": AMBER,
    "fusion": "#F472B6",
    "rerank": VIOLET,
    "generation": CYAN,
}


def inject_css():
    import streamlit as st

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(160deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] {{
            background: {NAVY_LIGHT};
            border-right: 1px solid rgba(0,212,255,0.15);
        }}
        h1, h2, h3, h4 {{
            color: {TEXT} !important;
            font-family: 'Segoe UI', sans-serif;
            letter-spacing: 0.3px;
        }}
        .tt7-hero {{
            background: linear-gradient(120deg, rgba(0,212,255,0.10), rgba(124,58,237,0.12));
            border: 1px solid rgba(0,212,255,0.25);
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 18px;
        }}
        .tt7-card {{
            background: {CARD_BG};
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 14px;
        }}
        .tt7-badge {{
            display:inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 6px;
        }}
        .tt7-pill-cyan   {{ background: rgba(0,212,255,0.15); color:{CYAN}; border:1px solid rgba(0,212,255,0.4);}}
        .tt7-pill-violet {{ background: rgba(124,58,237,0.18); color:{VIOLET}; border:1px solid rgba(124,58,237,0.4);}}
        .tt7-pill-green  {{ background: rgba(34,197,94,0.15); color:{GREEN}; border:1px solid rgba(34,197,94,0.4);}}
        .tt7-pill-amber  {{ background: rgba(245,158,11,0.15); color:{AMBER}; border:1px solid rgba(245,158,11,0.4);}}
        .tt7-chunk {{
            background: rgba(255,255,255,0.03);
            border-left: 3px solid {CYAN};
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            font-size: 13px;
            color: {MUTED};
        }}
        .tt7-stepline {{
            border-bottom: 1px dashed rgba(255,255,255,0.12);
            margin: 10px 0;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, {CYAN}, {VIOLET});
            color: #06080F;
            border: none;
            font-weight: 600;
            border-radius: 8px;
        }}
        .tt7-footer {{
            color: {MUTED};
            font-size: 12px;
            text-align: center;
            margin-top: 30px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    import streamlit as st

    st.markdown(
        f"""
        <div class="tt7-hero">
            <div style="font-size:12px;letter-spacing:2px;color:{CYAN};font-weight:700;">
                TECHTRINETRA7 &nbsp;•&nbsp; AWAKEN THE THIRD EYE OF TECH LEARNING
            </div>
            <h1 style="margin:6px 0 2px 0;">{title}</h1>
            <div style="color:{MUTED};font-size:14px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
