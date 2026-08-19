import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINE_PATH = "customer_churn_pipeline.pkl"


# =========================================================
# GLOBAL CSS
# =========================================================

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #070C14;
    --panel: #0B111B;
    --card: #0E1724;
    --card2: #101B2A;
    --border: #1E2C40;
    --text: #EAF0F8;
    --muted: #8EA0B8;
    --cyan: #42D8FF;
    --blue: #338DFF;
    --green: #45D88C;
    --green-dark: #19A96B;
    --red: #FF4D5A;
    --red-dark: #D93650;
    --purple: #9B6CFF;
}

* {
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            900px 500px at 35% -20%,
            rgba(46, 135, 255, 0.10),
            transparent 65%
        ),
        radial-gradient(
            700px 400px at 100% 20%,
            rgba(40, 216, 255, 0.045),
            transparent 70%
        ),
        var(--bg);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: #09101A;
    border-right: 1px solid #1A2637;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0.7rem;
}

section[data-testid="stSidebar"] .block-container {
    padding: 0.4rem 0.85rem 1.5rem 0.85rem;
}

/* Hide Streamlit branding/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Main page */
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

/* Brand */
.brand {
    margin-bottom: 1rem;
}

.brand-eyebrow {
    color: var(--cyan);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.brand-title {
    color: #F4F7FB;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.15rem;
    font-weight: 700;
    line-height: 1.05;
    margin-top: 0.2rem;
}

.brand-sub {
    color: #91A1B7;
    font-size: 0.78rem;
    margin-top: 0.4rem;
}

/* Sidebar */
.sidebar-brand {
    color: var(--cyan);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 0.35rem 0 0.7rem 0.1rem;
}

.section-label {
    color: #5F7794;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    padding-top: 0.62rem;
    margin-top: 0.45rem;
    margin-bottom: 0.35rem;
    border-top: 1px solid #182435;
}

/* KPI cards */
.kpi-card {
    position: relative;
    min-height: 86px;
    padding: 0.85rem 0.95rem;
    background:
        linear-gradient(
            135deg,
            rgba(18, 31, 49, 0.95),
            rgba(10, 20, 34, 0.95)
        );
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

.kpi-label {
    color: #72869F;
    font-size: 0.60rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.kpi-value {
    color: #F0F5FB;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
    line-height: 1.15;
    margin-top: 0.22rem;
}

.kpi-sub {
    color: #7D8EA5;
    font-size: 0.61rem;
    margin-top: 0.25rem;
}

.kpi-icon {
    position: absolute;
    right: 0.8rem;
    top: 50%;
    transform: translateY(-50%);
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    background: rgba(38, 142, 255, 0.15);
    border: 1px solid rgba(57, 150, 255, 0.12);
}

.kpi-icon.green {
    background: rgba(69, 216, 140, 0.13);
    color: var(--green);
}

.kpi-icon.purple {
    background: rgba(155, 108, 255, 0.14);
    color: #A87CFF;
}

/* Main cards */
.panel {
    background:
        linear-gradient(
            145deg,
            rgba(13, 24, 39, 0.98),
            rgba(8, 16, 28, 0.98)
        );
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

.panel-title {
    color: var(--cyan);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Gauge - CSS only.
   We intentionally do NOT use SVG here because Streamlit's HTML sanitizer
   can remove/alter inline SVG in some versions. */
.gauge-wrap {
    padding: 0.75rem 0.85rem 0.35rem 0.85rem;
    text-align: center;
}

.gauge-title {
    text-align: left;
    color: var(--cyan);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0 0 0.05rem 0.15rem;
}

.gauge-visual {
    position: relative;
    width: min(100%, 480px);
    height: 275px;
    margin: 0 auto;
    overflow: hidden;
}

.gauge-ring {
    position: absolute;
    left: 50%;
    top: 10px;
    width: 285px;
    height: 285px;
    transform: translateX(-50%);
    border-radius: 50%;
    background: #162334;
    -webkit-mask: radial-gradient(circle, transparent 0 83%, #000 84% 100%);
    mask: radial-gradient(circle, transparent 0 83%, #000 84% 100%);
}

.gauge-ring-color {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(
        from 270deg,
        #45D88C 0deg var(--safe-angle),
        #FF4D5A var(--safe-angle) 180deg,
        transparent 180deg 360deg
    );
}

.gauge-marker {
    position: absolute;
    width: 19px;
    height: 19px;
    left: var(--marker-x);
    top: var(--marker-y);
    transform: translate(-50%, -50%);
    border-radius: 50%;
    background: #F8FAFD;
    border: 1px solid #DCE5EF;
    box-shadow: 0 0 10px rgba(255,255,255,0.35);
    z-index: 5;
}

.gauge-center {
    position: absolute;
    left: 50%;
    top: 88px;
    transform: translateX(-50%);
    width: 190px;
    z-index: 4;
}

.gauge-percent {
    color: #FF334F;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.55rem;
    font-weight: 700;
    line-height: 1;
}

.gauge-center-label {
    color: #EAF0F8;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.25rem;
}

.gauge-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-top: 0.55rem;
    padding: 0.38rem 0.8rem;
    border-radius: 999px;
    color: var(--badge-color);
    background: var(--badge-bg);
    border: 1px solid var(--badge-border);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.66rem;
    font-weight: 700;
}

.gauge-side {
    position: absolute;
    top: 225px;
    z-index: 4;
    text-align: center;
}

.gauge-side.left { left: calc(50% - 145px); }
.gauge-side.right { right: calc(50% - 145px); }

.gauge-side-percent {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
}

.gauge-side-percent.green { color: #45D88C; }
.gauge-side-percent.red { color: #FF4D5A; }

.gauge-side-label {
    color: #8293A9;
    font-size: 0.57rem;
    margin-top: 0.18rem;
}

@media (max-width: 700px) {
    .gauge-visual { height: 250px; }
    .gauge-ring { width: 255px; height: 255px; }
    .gauge-side { top: 205px; }
    .gauge-side.left { left: calc(50% - 130px); }
    .gauge-side.right { right: calc(50% - 130px); }
    .gauge-center { top: 78px; }
}

.gauge-result {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0 0.05rem 0.15rem 0.05rem;
    padding: 0.72rem 0.85rem;
    min-height: 66px;
    border-radius: 9px;
    background: rgba(17, 78, 70, 0.22);
    border: 1px solid rgba(42, 196, 140, 0.23);
    text-align: left;
}

.gauge-result.high {
    background: rgba(103, 24, 40, 0.20);
    border-color: rgba(255, 77, 90, 0.25);
}

.gauge-result-icon {
    flex: 0 0 31px;
    width: 31px;
    height: 31px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.gauge-result-icon svg {
    width: 27px;
    height: 27px;
}

.gauge-result-copy {
    min-width: 0;
}

.gauge-result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--green);
}

.gauge-result.high .gauge-result-title {
    color: var(--red);
}

.gauge-result-text {
    color: #9AACBF;
    font-size: 0.61rem;
    margin-top: 0.18rem;
}

/* Breakdown */
.breakdown {
    padding: 0.8rem 0.85rem;
}

.factor {
    display: grid;
    grid-template-columns: 25px 1fr 48%;
    gap: 0.45rem;
    align-items: center;
    margin: 0.55rem 0;
}

.factor-icon {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--red);
    background: rgba(255, 77, 90, 0.10);
    border: 1px solid rgba(255, 77, 90, 0.12);
    font-size: 0.60rem;
}

.factor-icon.positive {
    color: var(--green);
    background: rgba(69, 216, 140, 0.10);
    border-color: rgba(69, 216, 140, 0.12);
}

.factor-name {
    color: #D8E0EA;
    font-size: 0.63rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.factor-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.factor-track {
    height: 6px;
    flex: 1;
    background: #111C2B;
    border-radius: 999px;
    overflow: hidden;
}

.factor-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #D93650, #FF5260);
}

.factor-fill.positive {
    background: linear-gradient(90deg, #29B975, #4BD990);
}

.factor-value {
    color: #D7DEE8;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.59rem;
    min-width: 38px;
    text-align: right;
}

.info-note {
    margin-top: 0.85rem;
    padding: 0.58rem 0.7rem;
    border: 1px solid #23344C;
    background: rgba(21, 40, 67, 0.35);
    border-radius: 8px;
    color: #8EA4BE;
    font-size: 0.60rem;
}

/* Profile */
.profile {
    padding: 0.72rem 0.85rem;
}

.profile-title {
    color: var(--cyan);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}

.profile-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.7rem;
}

.profile-item {
    display: grid;
    grid-template-columns: 31px 1fr;
    gap: 0.45rem;
    align-items: center;
    min-width: 0;
}

.profile-icon {
    width: 31px;
    height: 31px;
    border-radius: 9px;
    background: rgba(37, 126, 222, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #4DBDFF;
    font-size: 0.82rem;
}

.profile-icon.purple {
    color: #B07CFF;
    background: rgba(139, 91, 255, 0.13);
}

.profile-label {
    color: #7F91A9;
    font-size: 0.56rem;
}

.profile-value {
    color: #DDE5EE;
    font-size: 0.61rem;
    margin-top: 0.1rem;
}

/* Sidebar button */
.stButton > button {
    width: 100%;
    min-height: 38px;
    border: 0;
    border-radius: 8px;
    background: linear-gradient(135deg, #38D7FF, #716BFF);
    color: #061019;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    box-shadow: 0 8px 24px rgba(62, 142, 255, 0.15);
}

.stButton > button:hover {
    filter: brightness(1.08);
    border: 0;
    color: #061019;
}

/* Inputs */
div[data-baseweb="select"] > div,
.stNumberInput input {
    background: #0C1522 !important;
    border-color: #1E2B3D !important;
    color: #EAF0F8 !important;
    border-radius: 7px !important;
}

div[data-baseweb="select"] {
    font-size: 0.75rem !important;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: #8EA0B8 !important;
    font-size: 0.69rem !important;
}

.stSlider [data-baseweb="slider"] {
    margin-top: -0.2rem;
}

/* Responsive */
@media (max-width: 1050px) {
    .brand-title {
        font-size: 1.75rem;
    }

    .profile-grid {
        grid-template-columns: repeat(3, 1fr);
    }

    .factor {
        grid-template-columns: 25px 1fr 42%;
    }
}

@media (max-width: 700px) {
    .profile-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# HTML HELPER
# =========================================================

def html(markup: str):
    markup = markup.strip()

    # st.html avoids Streamlit Markdown interpreting indented
    # SVG/HTML as a code block.
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


try:
    pipeline = load_pipeline()
    model_ready = True
    model_error = None
except FileNotFoundError:
    pipeline = None
    model_ready = False
    model_error = (
        f"{PIPELINE_PATH} was not found. "
        "Place it in the same directory as app.py."
    )
except Exception as exc:
    pipeline = None
    model_ready = False
    model_error = f"Model loading failed: {exc}"


# =========================================================
# RISK GAUGE
# =========================================================

def churn_panel_html(probability: float, high_risk: bool) -> str:
    """Render the complete churn card inside a real HTML iframe.

    Using st.components.v1.html avoids Streamlit HTML sanitization/layout
    differences that can hide CSS/SVG content inside st.html().
    """
    p = float(np.clip(probability, 0.0, 1.0))
    safe = 1.0 - p

    # Geometry for a true upper semicircle.
    cx, cy, r = 210.0, 205.0, 145.0
    # IMPORTANT: the white separator is the boundary between the
    # green safe portion and red churn portion.
    #
    # The gauge runs from LEFT (0% churn) to RIGHT (100% churn).
    # Therefore the boundary is positioned after the SAFE fraction
    # (1 - p) of the semicircle, which gives an angle of p * 180°.
    # Example: p=0.403 -> the marker is slightly RIGHT of the top.
    angle_deg = p * 180.0
    angle = np.radians(angle_deg)
    marker_x = cx + r * np.cos(angle)
    marker_y = cy - r * np.sin(angle)

    # 40.3% in the reference is still shown as At Risk,
    # even though classification stays below 50%.
    badge_at_risk = p >= 0.30
    badge_text = "At Risk" if badge_at_risk else "Safe"
    badge_color = "#FF4D5A" if badge_at_risk else "#45D88C"

    result_color = "#FF4D5A" if high_risk else "#45D88C"
    result_bg = "rgba(17,78,70,.22)" if not high_risk else "rgba(103,24,40,.22)"
    result_border = "rgba(42,196,140,.23)" if not high_risk else "rgba(255,77,90,.25)"

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    * {{ box-sizing: border-box; }}
    html, body {{
        margin: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: transparent;
        font-family: Inter, Arial, sans-serif;
    }}
    .card {{
        width: 100%;
        min-height: 100%;
        padding: 10px 14px 12px 14px;
        border: 1px solid #1E2C40;
        border-radius: 10px;
        background: linear-gradient(145deg, rgba(13,24,39,.98), rgba(8,16,28,.98));
        color: #EAF0F8;
    }}
    .title {{
        color: #42D8FF;
        font-family: 'Space Grotesk', Inter, Arial, sans-serif;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 0 0 2px;
    }}
    .gauge-box {{
        position: relative;
        width: 100%;
        height: 292px;
        margin-top: -4px;
    }}
    svg {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        display: block;
    }}
    .result {{
        margin-top: -1px;
        padding: 12px 14px;
        border-radius: 10px;
        background: {result_bg};
        border: 1px solid {result_border};
        display: flex;
        gap: 11px;
        align-items: center;
    }}
    .shield {{
        width: 24px;
        height: 24px;
        flex: 0 0 24px;
    }}
    .result-title {{
        color: {result_color};
        font-family: 'Space Grotesk', Inter, Arial, sans-serif;
        font-size: 13px;
        font-weight: 700;
        line-height: 1.2;
    }}
    .result-text {{
        color: #9AACBF;
        font-size: 11px;
        margin-top: 4px;
    }}
</style>
</head>
<body>
<div class="card">
    <div class="title">CHURN PROBABILITY</div>

    <div class="gauge-box">
        <svg viewBox="0 0 420 292" preserveAspectRatio="xMidYMid meet">
            <defs>
                <linearGradient id="greenGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#3ED782"/>
                    <stop offset="100%" stop-color="#54E19A"/>
                </linearGradient>
                <linearGradient id="redGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stop-color="#FF4455"/>
                    <stop offset="100%" stop-color="#FF625D"/>
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="4" result="b"/>
                    <feMerge>
                        <feMergeNode in="b"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>

            <!-- Track -->
            <path d="M65 205 A145 145 0 0 1 355 205"
                  fill="none" stroke="#162334" stroke-width="15"
                  stroke-linecap="round"/>

            <!-- Full green base -->
            <path d="M65 205 A145 145 0 0 1 355 205"
                  fill="none" stroke="url(#greenGrad)" stroke-width="15"
                  stroke-linecap="round"/>

            <!-- Red risk section, proportional to churn probability -->
            <path d="M65 205 A145 145 0 0 1 355 205"
                  fill="none" stroke="url(#redGrad)" stroke-width="15"
                  stroke-linecap="round"
                  pathLength="100"
                  stroke-dasharray="{p*100:.4f} 100"
                  stroke-dashoffset="{-safe*100:.4f}"/>

            <!-- Center marker at probability boundary -->
            <circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="10"
                    fill="#F8FAFD" stroke="#DCE5EF" stroke-width="2"
                    filter="url(#glow)"/>

            <!-- Main probability -->
            <text x="210" y="155" text-anchor="middle"
                  fill="#FF334F"
                  font-family="Space Grotesk, Inter, sans-serif"
                  font-size="43" font-weight="700">{p*100:.1f}%</text>

            <text x="210" y="181" text-anchor="middle"
                  fill="#EAF0F8"
                  font-family="Inter, Arial, sans-serif"
                  font-size="13" font-weight="500">Churn Probability</text>

            <!-- Badge -->
            <rect x="169" y="194" width="82" height="29" rx="15"
                  fill="{badge_color}" fill-opacity="0.16"/>
            <text x="210" y="213" text-anchor="middle"
                  fill="{badge_color}"
                  font-family="Space Grotesk, Inter, sans-serif"
                  font-size="12" font-weight="700">{badge_text}</text>

            <!-- Bottom labels -->
            <text x="75" y="253" text-anchor="middle"
                  fill="#45D88C"
                  font-family="Space Grotesk, Inter, sans-serif"
                  font-size="14" font-weight="700">{safe*100:.1f}%</text>
            <text x="75" y="272" text-anchor="middle"
                  fill="#8293A9"
                  font-family="Inter, Arial, sans-serif"
                  font-size="10">Unlikely to churn</text>

            <text x="345" y="253" text-anchor="middle"
                  fill="#FF4D5A"
                  font-family="Space Grotesk, Inter, sans-serif"
                  font-size="14" font-weight="700">{p*100:.1f}%</text>
            <text x="345" y="272" text-anchor="middle"
                  fill="#8293A9"
                  font-family="Inter, Arial, sans-serif"
                  font-size="10">Likely to churn</text>
        </svg>
    </div>

    <div class="result">
        <svg class="shield" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
            <path d="M16 3.5 L27 8 V15.8 C27 22.2 22.7 26.8 16 29 C9.3 26.8 5 22.2 5 15.8 V8 Z"
                  fill="none" stroke="{result_color}" stroke-width="1.8"
                  stroke-linejoin="round"/>
            <path d="M10.5 16.2 L14.2 19.7 L21.8 11.9"
                  fill="none" stroke="{result_color}" stroke-width="2"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div>
            <div class="result-title">
                {'Customer is likely to churn' if high_risk else 'Customer is unlikely to churn'}
            </div>
            <div class="result-text">
                The model estimates a {p*100:.1f}% probability of churn.
            </div>
        </div>
    </div>
</div>
</body>
</html>
"""


# =========================================================
# MODEL PREDICTION
# =========================================================

def predict_customer(input_data):
    prediction = pipeline.predict(input_data)[0]

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_data)[0]

        if hasattr(pipeline, "classes_"):
            classes = list(pipeline.classes_)
            if 1 in classes:
                churn_index = classes.index(1)
            elif len(classes) == 2:
                churn_index = 1
            else:
                churn_index = int(np.argmax(probabilities))
        else:
            churn_index = 1 if len(probabilities) > 1 else 0

        churn_probability = float(probabilities[churn_index])
    else:
        churn_probability = 1.0 if int(prediction) == 1 else 0.0

    churn_probability = float(np.clip(churn_probability, 0, 1))

    return (
        int(prediction),
        churn_probability,
        1.0 - churn_probability,
    )


# =========================================================
# OPTIONAL MODEL COEFFICIENT BREAKDOWN
# =========================================================

def get_model_contributions(input_data):
    """
    Attempts to calculate real Logistic Regression feature
    contributions. If the saved pipeline does not expose the
    expected preprocessing/model structure, returns None.

    These are log-odds contributions, NOT percentages.
    """

    try:
        # Typical sklearn Pipeline:
        # preprocessing -> LogisticRegression
        if not hasattr(pipeline, "named_steps"):
            return None

        steps = pipeline.named_steps

        model = None
        preprocessor = None

        for name, step in steps.items():
            if hasattr(step, "coef_"):
                model = step
            if hasattr(step, "transform") and hasattr(step, "get_feature_names_out"):
                preprocessor = step

        if model is None or preprocessor is None:
            return None

        transformed = preprocessor.transform(input_data)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        coefficients = np.asarray(model.coef_)

        if coefficients.ndim != 2 or coefficients.shape[0] != 1:
            return None

        coefficients = coefficients[0]

        names = list(preprocessor.get_feature_names_out())

        if len(names) != len(coefficients):
            return None

        values = transformed[0]

        contributions = values * coefficients

        result = pd.DataFrame(
            {
                "feature": names,
                "contribution": contributions,
            }
        )

        result = result.sort_values(
            "contribution",
            key=lambda s: s.abs(),
            ascending=False,
        )

        return result.head(6)

    except Exception:
        return None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    html("""
<div class="sidebar-brand">TELCO ANALYTICS</div>
""")

    html('<div class="section-label">Account</div>')

    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    html('<div class="section-label">Charges</div>')

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=1.0,
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=20000.0,
        value=600.0,
        step=10.0,
    )

    html('<div class="section-label">Demographics</div>')

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"],
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"],
    )

    html('<div class="section-label">Phone & Internet</div>')

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"],
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"],
    )

    html('<div class="section-label">Additional Services</div>')

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"],
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"],
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"],
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"],
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"],
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"],
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "Predict Churn",
        use_container_width=True,
        type="primary",
    )


# =========================================================
# INPUT DATA
# =========================================================

input_data = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    ]
)


# =========================================================
# HEADER
# =========================================================

html("""
<div class="brand">
    <div class="brand-eyebrow">TELCO ANALYTICS</div>
    <div class="brand-title">Customer Churn Prediction</div>
    <div class="brand-sub">
        Enter a customer profile and use the trained machine learning
        pipeline to estimate churn probability.
    </div>
</div>
""")


# =========================================================
# KPI CARDS
# =========================================================

active_services = sum(
    service == "Yes"
    for service in [
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
    ]
)

k1, k2, k3, k4 = st.columns(4, gap="small")

with k1:
    html(f"""
<div class="kpi-card">
    <div class="kpi-label">TENURE</div>
    <div class="kpi-value">{tenure} months</div>
    <div class="kpi-sub">Customer tenure</div>
    <div class="kpi-icon">▣</div>
</div>
""")

with k2:
    html(f"""
<div class="kpi-card">
    <div class="kpi-label">MONTHLY CHARGES</div>
    <div class="kpi-value">${monthly_charges:,.2f}</div>
    <div class="kpi-sub">Recurring monthly</div>
    <div class="kpi-icon">$</div>
</div>
""")

with k3:
    html(f"""
<div class="kpi-card">
    <div class="kpi-label">CONTRACT</div>
    <div class="kpi-value">{contract}</div>
    <div class="kpi-sub">Current contract</div>
    <div class="kpi-icon purple">▤</div>
</div>
""")

with k4:
    html(f"""
<div class="kpi-card">
    <div class="kpi-label">SERVICES ACTIVE</div>
    <div class="kpi-value">{active_services} / 6</div>
    <div class="kpi-sub">Active services</div>
    <div class="kpi-icon green">◆</div>
</div>
""")


st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION AREA
# =========================================================

if predict_clicked and model_ready:

    try:
        prediction, churn_probability, stay_probability = predict_customer(
            input_data
        )

        high_risk = churn_probability >= 0.5

        left, right = st.columns(
            [1.05, 0.95],
            gap="small",
        )

        # -----------------------------------------------------
        # LEFT: GAUGE
        # -----------------------------------------------------

        with left:
            components.html(
                churn_panel_html(churn_probability, high_risk),
                height=410,
                scrolling=False,
            )

        # -----------------------------------------------------
        # RIGHT: BREAKDOWN
        # -----------------------------------------------------

        with right:

            contributions = get_model_contributions(input_data)

            if contributions is not None and len(contributions) > 0:

                factors = []

                max_abs = max(
                    float(contributions["contribution"].abs().max()),
                    1e-9,
                )

                for _, row in contributions.iterrows():

                    feature = str(row["feature"])
                    value = float(row["contribution"])

                    clean_name = (
                        feature
                        .replace("onehot__", "")
                        .replace("num__", "")
                        .replace("cat__", "")
                        .replace("remainder__", "")
                        .replace("_", " ")
                    )

                    clean_name = clean_name.title()

                    positive = value > 0
                    width = min(abs(value) / max_abs * 100, 100)

                    factors.append(
                        f"""
<div class="factor">
    <div class="factor-icon {'positive' if not positive else ''}">
        {'−' if not positive else '+'}
    </div>

    <div class="factor-name" title="{clean_name}">
        {clean_name}
    </div>

    <div class="factor-bar-wrap">
        <div class="factor-track">
            <div
                class="factor-fill {'positive' if not positive else ''}"
                style="width:{width:.1f}%"
            ></div>
        </div>

        <div class="factor-value">
            {value:+.2f}
        </div>
    </div>
</div>
"""
                    )

                factor_html = "".join(factors)

                html(f"""
<div class="panel">
    <div class="breakdown">
        <div class="panel-title">PREDICTION BREAKDOWN</div>

        {factor_html}

        <div class="info-note">
            Positive values increase churn risk; negative values decrease it.
            Values shown are model contribution scores, not percentages.
        </div>
    </div>
</div>
""")

            else:

                # Fallback that keeps the same professional visual
                # when the saved model doesn't expose coefficients.
                fallback = [
                    ("Contract Type", contract, 75, True),
                    ("Tenure", f"{tenure} months", 60, False),
                    ("Monthly Charges", f"${monthly_charges:.2f}", 48, True),
                    ("Paperless Billing", paperless_billing, 34, True),
                    ("Payment Method", payment_method, 29, True),
                    ("Online Security", online_security, 23, True),
                ]

                rows = []

                for name, value, width, positive in fallback:
                    rows.append(
                        f"""
<div class="factor">
    <div class="factor-icon {'positive' if not positive else ''}">
        {'+' if positive else '−'}
    </div>

    <div class="factor-name">
        {name} ({value})
    </div>

    <div class="factor-bar-wrap">
        <div class="factor-track">
            <div
                class="factor-fill {'positive' if not positive else ''}"
                style="width:{width}%"
            ></div>
        </div>
        <div class="factor-value">
            {'Risk' if positive else 'Protect'}
        </div>
    </div>
</div>
"""
                    )

                html(f"""
<div class="panel">
    <div class="breakdown">
        <div class="panel-title">PREDICTION BREAKDOWN</div>

        {''.join(rows)}

        <div class="info-note">
            The saved pipeline does not expose feature coefficients,
            so detailed model contribution scores are unavailable.
        </div>
    </div>
</div>
""")


        # -----------------------------------------------------
        # CUSTOMER PROFILE SUMMARY
        # -----------------------------------------------------

        profile_items = [
            ("◉", "Internet Service", f"{internet_service} internet", ""),
            ("⌕", "Multiple Lines", multiple_lines, ""),
            ("♙", "Dependents", dependents, ""),
            ("▣", "Streaming TV", streaming_tv, ""),
            ("◉", "Tech Support", tech_support, "purple"),
            ("◇", "Device Protection", device_protection, "purple"),
        ]

        profile_html = []

        for icon, label, value, icon_class in profile_items:
            profile_html.append(
                f"""
<div class="profile-item">
    <div class="profile-icon {icon_class}">{icon}</div>
    <div>
        <div class="profile-label">{label}</div>
        <div class="profile-value">{value}</div>
    </div>
</div>
"""
            )

        html(f"""
<div class="panel" style="margin-top:8px;">
    <div class="profile">
        <div class="profile-title">CUSTOMER PROFILE SUMMARY</div>

        <div class="profile-grid">
            {''.join(profile_html)}
        </div>
    </div>
</div>
""")

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

elif not model_ready:

    st.error(model_error)

else:

    html("""
<div class="panel" style="padding:1.2rem;">
    <div class="panel-title">READY</div>
    <div style="
        color:#8EA0B8;
        font-size:0.75rem;
        margin-top:0.35rem;
    ">
        Enter a customer profile from the sidebar and click
        Predict Churn.
    </div>
</div>
""")


# =========================================================
# FOOTER
# =========================================================

html("""
<div style="
    color:#53647A;
    font-size:0.57rem;
    margin-top:0.55rem;
    text-align:right;
">
    Logistic Regression Pipeline · StandardScaler + OneHotEncoder ·
    Telco Customer Churn
</div>
""")