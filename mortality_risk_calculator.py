"""
Elderly ICU Mortality Risk Calculator – Version 1.0 (Sept 2025)
Award-Grade Streamlit Application | Cox Proportional Hazard Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Mortality Risk Calculator",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Deep Navy + Electric Teal medical design language
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
    --navy:        #05101f;
    --navy-mid:    #0a1f38;
    --navy-light:  #102d50;
    --teal:        #00c9b1;
    --teal-dim:    #00a898;
    --teal-glow:   rgba(0,201,177,0.15);
    --amber:       #f59e0b;
    --rose:        #f43f5e;
    --violet:      #8b5cf6;
    --slate:       #8fa3b8;
    --white:       #f0f6ff;
    --card-bg:     rgba(10,31,56,0.75);
    --border:      rgba(0,201,177,0.18);
    --font-head:   'Space Grotesk', sans-serif;
    --font-body:   'DM Sans', sans-serif;
    --font-mono:   'JetBrains Mono', monospace;
}

/* ── Global reset ── */
html, body, .stApp { background: var(--navy) !important; color: var(--white) !important; font-family: var(--font-body) !important; }
.stApp { min-height: 100vh; background: radial-gradient(ellipse 120% 80% at 10% -10%, #0d2a4a 0%, #05101f 60%) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1300px !important; }

/* ── Typography ── */
h1,h2,h3,h4,h5 { font-family: var(--font-head) !important; color: var(--white) !important; }

/* ── Streamlit widget overrides ── */
/* Labels */
.stNumberInput label, .stSelectbox label, .stRadio label,
.stSlider label, .stCheckbox label, .stTextInput label {
    color: var(--teal) !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

/* Input boxes */
.stNumberInput input, .stTextInput input {
    background: rgba(16,45,80,0.9) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: var(--font-mono) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--teal) !important;
    box-shadow: 0 0 0 3px var(--teal-glow) !important;
    outline: none !important;
}
.stNumberInput input::placeholder { color: rgba(143,163,184,0.5) !important; }

/* Select boxes */
.stSelectbox > div > div {
    background: rgba(16,45,80,0.9) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}
.stSelectbox > div > div:focus-within { border-color: var(--teal) !important; }
[data-baseweb="select"] * { color: #ffffff !important; background: #0a1f38 !important; }

/* Radio buttons */
.stRadio > div { gap: 10px !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--white) !important; font-size: 14px !important; }
.stRadio label { color: var(--white) !important; font-size: 14px !important; text-transform: none !important; letter-spacing: 0 !important; font-weight: 400 !important; }
div[data-testid="stRadio"] label { background: rgba(16,45,80,0.6) !important; border: 1.5px solid var(--border) !important; border-radius: 8px !important; padding: 7px 16px !important; cursor: pointer !important; transition: all 0.2s !important; }
div[data-testid="stRadio"] label:has(input:checked) { background: var(--teal-glow) !important; border-color: var(--teal) !important; }

/* Checkboxes */
.stCheckbox label { color: var(--white) !important; font-size: 13px !important; font-weight: 400 !important; text-transform: none !important; letter-spacing: 0 !important; }
.stCheckbox [data-testid="stMarkdownContainer"] p { color: var(--slate) !important; }

/* Buttons */
.stButton > button {
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.05em !important;
    border-radius: 12px !important;
    border: none !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00c9b1 0%, #0096d6 100%) !important;
    color: var(--navy) !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 20px rgba(0,201,177,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,201,177,0.50) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1.5px solid var(--border) !important;
    color: var(--slate) !important;
    padding: 10px 24px !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--teal) !important;
    color: var(--teal) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-head) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    color: var(--slate) !important;
    background: transparent !important;
    border: none !important;
    padding: 12px 22px !important;
    border-radius: 8px 8px 0 0 !important;
    text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] {
    color: var(--teal) !important;
    background: var(--teal-glow) !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--white) !important;
    font-family: var(--font-head) !important;
    font-size: 13px !important;
}
.streamlit-expanderContent { background: rgba(5,16,31,0.5) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 10px 10px !important; }

/* Dividers */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* Info / success / error boxes */
.stAlert { border-radius: 10px !important; border: none !important; }
[data-testid="stNotificationContentInfo"] { background: rgba(0,201,177,0.1) !important; color: var(--teal) !important; }
[data-testid="stNotificationContentError"] { background: rgba(244,63,94,0.1) !important; }
[data-testid="stNotificationContentSuccess"] { background: rgba(16,185,129,0.1) !important; }

/* Dataframe */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stDataFrameResizable"] { background: var(--navy-mid) !important; }

/* Slider */
.stSlider [data-testid="stTickBar"] { color: var(--slate) !important; }
.stSlider [role="slider"] { background: var(--teal) !important; }

/* ── Custom components ── */
.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 0 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.nav-logo {
    display: flex; align-items: center; gap: 12px;
}
.nav-logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #00c9b1, #0096d6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.nav-title { font-family: var(--font-head); font-size: 18px; font-weight: 700; color: var(--white); }
.nav-sub { font-size: 11px; color: var(--slate); letter-spacing: 0.08em; text-transform: uppercase; }
.nav-badge {
    background: var(--teal-glow); border: 1px solid var(--teal);
    border-radius: 20px; padding: 5px 14px;
    font-size: 11px; font-family: var(--font-mono);
    color: var(--teal); letter-spacing: 0.05em;
}

.step-indicator {
    display: flex; align-items: center; gap: 0; margin-bottom: 28px;
}
.step {
    display: flex; align-items: center; gap: 10px;
    font-family: var(--font-head); font-size: 13px; font-weight: 600;
    color: var(--slate); white-space: nowrap;
}
.step.active { color: var(--teal); }
.step.done { color: #10b981; }
.step-num {
    width: 28px; height: 28px; border-radius: 50%;
    border: 2px solid currentColor;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0;
}
.step-line { flex: 1; height: 1px; background: var(--border); margin: 0 12px; min-width: 30px; }

.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.card-sm { padding: 18px 20px; border-radius: 12px; }

.section-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--teal-glow); border: 1px solid rgba(0,201,177,0.3);
    border-radius: 20px; padding: 5px 14px;
    font-family: var(--font-head); font-size: 11px; font-weight: 700;
    color: var(--teal); letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 14px;
}

.metric-stat {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 22px;
    position: relative; overflow: hidden;
}
.metric-stat::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--teal), transparent);
}
.metric-stat .label { font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--slate); margin-bottom: 8px; }
.metric-stat .value { font-family: var(--font-head); font-size: 32px; font-weight: 700; color: var(--white); line-height: 1; }
.metric-stat .sub { font-size: 12px; color: var(--slate); margin-top: 6px; }

.risk-banner {
    border-radius: 16px; padding: 26px 30px;
    border: 1px solid; position: relative; overflow: hidden;
}
.risk-banner::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, currentColor 0%, transparent 70%);
    opacity: 0.06;
}

.risk-label-chip {
    display: inline-flex; align-items: center; gap: 8px;
    border-radius: 30px; padding: 8px 22px;
    font-family: var(--font-head); font-size: 15px; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
}

.input-group-title {
    font-family: var(--font-head); font-size: 13px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--teal); padding: 10px 0 6px 0;
    border-bottom: 1px solid var(--border); margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}

.cci-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }

.results-header {
    font-family: var(--font-head); font-size: 28px; font-weight: 700;
    line-height: 1.2; margin-bottom: 6px;
}
.results-sub { font-size: 14px; color: var(--slate); margin-bottom: 24px; }

.teal { color: var(--teal) !important; }
.mono { font-family: var(--font-mono) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BETA = {
    "cardiac_arrest":  2.7268,
    "log_inr":         1.7000,
    "mechanical_vent": 0.9076,
    "log_age":         1.55229,
    "log_cci":         0.8597,
    "cfs":             0.05976,
    "hematocrit":     -0.05827,
    "chloride":       -0.00096,
}

BASELINE_HAZARD = {
    1:0.000007, 2:0.000015, 3:0.000019, 4:0.000038, 5:0.000082,
    6:0.000099, 7:0.000109, 8:0.000132, 9:0.000145, 10:0.000159,
    11:0.000179, 12:0.000200, 13:0.000200, 14:0.000200, 15:0.000248,
    16:0.000275, 17:0.000334, 18:0.000367, 19:0.000403, 20:0.000442,
    21:0.000442, 22:0.000491, 24:0.000561, 26:0.000652, 30:0.000756,
}

CCI_CONDITIONS = {
    "Myocardial infarction": 1, "Congestive heart failure": 1,
    "Peripheral vascular disease": 1, "Cerebrovascular accident / TIA": 1,
    "Dementia": 1, "Chronic pulmonary disease": 1,
    "Connective tissue disease": 1, "Peptic ulcer disease": 1,
    "Mild liver disease": 1, "Uncomplicated diabetes": 1,
    "Hemiplegia": 2, "Moderate to severe CKD": 2,
    "Diabetes with end-organ damage": 2, "Localized solid tumor": 2,
    "Leukemia": 2, "Lymphoma": 2,
    "Moderate to severe liver disease": 3,
    "Metastatic solid tumor": 6, "AIDS": 6,
}

CFS_OPTIONS = {
    1: "1 — Very Fit", 2: "2 — Well", 3: "3 — Managing Well",
    4: "4 — Vulnerable", 5: "5 — Mildly Frail", 6: "6 — Moderately Frail",
    7: "7 — Severely Frail", 8: "8 — Very Severely Frail",
}

RISK_CONFIG = {
    "LOW":       {"band":"< 10%",  "color":"#10b981","bg":"rgba(16,185,129,0.08)","border":"rgba(16,185,129,0.4)","dot":"#10b981",
                  "rec":"ICU admission strongly recommended. Anticipate favourable outcome with standard monitoring protocols."},
    "MODERATE":  {"band":"10–30%", "color":"#f59e0b","bg":"rgba(245,158,11,0.08)","border":"rgba(245,158,11,0.4)","dot":"#f59e0b",
                  "rec":"ICU admission appropriate. Ensure early rehabilitation, daily goal-setting, and proactive family communication."},
    "HIGH":      {"band":"30–60%", "color":"#f43f5e","bg":"rgba(244,63,94,0.08)","border":"rgba(244,63,94,0.4)","dot":"#f43f5e",
                  "rec":"Discuss in triage rounds. Initiate family counselling, advance care planning, and palliative care consultation."},
    "VERY HIGH": {"band":"> 60%",  "color":"#c026d3","bg":"rgba(192,38,211,0.08)","border":"rgba(192,38,211,0.4)","dot":"#c026d3",
                  "rec":"Case-by-case ICU consideration. Strong recommendation for goals-of-care discussion and palliative care involvement."},
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def classify_risk(pct):
    if pct < 10:    return "LOW"
    elif pct <= 30: return "MODERATE"
    elif pct <= 60: return "HIGH"
    else:           return "VERY HIGH"

def get_mort_at(exp_score, day):
    if day in BASELINE_HAZARD:
        return (1 - np.exp(-BASELINE_HAZARD[day] * exp_score)) * 100
    sd = sorted(BASELINE_HAZARD.keys())
    for i in range(len(sd)-1):
        d1,d2 = sd[i],sd[i+1]
        if d1 <= day <= d2:
            a = (day-d1)/(d2-d1)
            h = BASELINE_HAZARD[d1] + a*(BASELINE_HAZARD[d2]-BASELINE_HAZARD[d1])
            return (1 - np.exp(-h * exp_score)) * 100
    return (1 - np.exp(-BASELINE_HAZARD[30] * exp_score)) * 100

def compute(age, ca, inr, mv, cci, cfs, hct, cl):
    la = np.log(max(age,1)); li = np.log(max(inr,0.001)); lc = np.log(max(cci,0.001))
    composite = (BETA["cardiac_arrest"]*ca + BETA["log_inr"]*li + BETA["mechanical_vent"]*mv +
                 BETA["log_age"]*la + BETA["log_cci"]*lc + BETA["cfs"]*cfs +
                 BETA["hematocrit"]*hct + BETA["chloride"]*cl)
    exp_s = np.exp(composite)
    breakdown = {
        "Cardiac Arrest":          BETA["cardiac_arrest"]*ca,
        "Loge INR":                BETA["log_inr"]*li,
        "Mechanical Ventilation":  BETA["mechanical_vent"]*mv,
        "Loge Age":                BETA["log_age"]*la,
        "Loge CCI":                BETA["log_cci"]*lc,
        "Clinical Frailty Scale":  BETA["cfs"]*cfs,
        "Hematocrit":              BETA["hematocrit"]*hct,
        "Chloride":                BETA["chloride"]*cl,
    }
    return composite, exp_s, breakdown

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">
    <div class="nav-logo-icon">🫀</div>
    <div>
      <div class="nav-title">ICU Mortality Risk Calculator</div>
      <div class="nav-sub">Elderly Patient · Cox Proportional Hazard Model</div>
    </div>
  </div>
  <div class="nav-badge">v1.0 · Sept 2025</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME / WELCOME
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    # Hero
    col_hero, col_ref = st.columns([3, 2], gap="large")

    with col_hero:
        st.markdown("""
        <div style="padding: 10px 0 24px 0;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:13px; font-weight:700;
                        letter-spacing:0.12em; text-transform:uppercase; color:#00c9b1; margin-bottom:12px;">
                CLINICAL DECISION SUPPORT TOOL
            </div>
            <h1 style="font-family:'Space Grotesk',sans-serif; font-size:42px; font-weight:700;
                       line-height:1.15; margin:0 0 16px 0; color:#f0f6ff;">
                Predict ICU Mortality<br>
                <span style="color:#00c9b1;">With Precision.</span>
            </h1>
            <p style="font-size:16px; color:#8fa3b8; line-height:1.7; max-width:520px; margin-bottom:28px;">
                A validated prognostic tool for clinicians managing elderly ICU admissions (≥ 65 years).
                Computes 7, 14, and 30-day mortality risk using the Cox Proportional Hazard model,
                with instant risk stratification and evidence-based clinical guidance.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("🔬  Begin Patient Assessment", type="primary", use_container_width=True):
                st.session_state.page = "calculator"
                st.rerun()
        with c2:
            if st.button("📖  How It Works", use_container_width=True):
                st.session_state.page = "about"
                st.rerun()

        # Feature chips
        st.markdown("""
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:24px;">
            <div style="background:rgba(0,201,177,0.08);border:1px solid rgba(0,201,177,0.25);
                        border-radius:20px;padding:6px 14px;font-size:12px;color:#00c9b1;font-weight:600;">
                ⚡ Instant Results
            </div>
            <div style="background:rgba(0,150,214,0.08);border:1px solid rgba(0,150,214,0.25);
                        border-radius:20px;padding:6px 14px;font-size:12px;color:#60b8e8;font-weight:600;">
                📊 Survival Curves
            </div>
            <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);
                        border-radius:20px;padding:6px 14px;font-size:12px;color:#a78bfa;font-weight:600;">
                🏥 Clinical Guidance
            </div>
            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                        border-radius:20px;padding:6px 14px;font-size:12px;color:#fcd34d;font-weight:600;">
                🔢 CCI Calculator Built-in
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ref:
        st.markdown("""
        <div class="card" style="margin-top:10px;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;
                        color:#00c9b1;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">
                📋 Risk Band Reference
            </div>
        """, unsafe_allow_html=True)

        for label, cfg in RISK_CONFIG.items():
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 0;
                        border-bottom:1px solid rgba(0,201,177,0.1);">
                <div style="width:10px;height:10px;border-radius:50%;background:{cfg['color']};
                            margin-top:4px;flex-shrink:0;box-shadow:0 0 8px {cfg['color']};"></div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
                                font-weight:700;color:{cfg['color']};">
                        {label} <span style="color:#8fa3b8;font-weight:400;font-size:12px;">({cfg['band']})</span>
                    </div>
                    <div style="font-size:12px;color:#8fa3b8;margin-top:3px;line-height:1.5;">{cfg['rec']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Stats row
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:8px;">
        <div class="metric-stat card-sm">
            <div class="label">Model Variables</div>
            <div class="value teal">8</div>
            <div class="sub">Clinical parameters</div>
        </div>
        <div class="metric-stat card-sm">
            <div class="label">Time Points</div>
            <div class="value teal">25</div>
            <div class="sub">Days 1 – 30</div>
        </div>
        <div class="metric-stat card-sm">
            <div class="label">Age Threshold</div>
            <div class="value teal">≥ 65</div>
            <div class="sub">Years old</div>
        </div>
        <div class="metric-stat card-sm">
            <div class="label">Risk Categories</div>
            <div class="value teal">4</div>
            <div class="sub">LOW → VERY HIGH</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CALCULATOR — 3-step tabbed layout
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "calculator":

    # Back button
    col_back, _ = st.columns([1, 6])
    with col_back:
        if st.button("← Back", key="back_calc"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:#f0f6ff;">
            Patient Assessment Form
        </div>
        <div style="font-size:14px;color:#8fa3b8;margin-top:4px;">
            Complete all sections below and click <strong style="color:#00c9b1;">Calculate Risk</strong> to generate the mortality assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  🧑  PATIENT DETAILS  ", "  🩺  COMORBIDITIES (CCI)  ", "  📊  RESULTS  "])

    # ─── TAB 1: Patient Details ───────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown('<div class="input-group-title">🧑 Demographics</div>', unsafe_allow_html=True)

            age_val = st.number_input(
                "Age upon ICU admission (years)",
                min_value=65, max_value=120, step=1,
                value=st.session_state.get("age_val", None),
                placeholder="Enter age (≥ 65)",
                key="age_input",
                help="Patient must be ≥ 65 years. Age also contributes to the CCI score automatically."
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="input-group-title">⚡ Acute Clinical Events</div>', unsafe_allow_html=True)

            cardiac_arrest_input = st.radio(
                "Cardiac arrest prior to ICU admission?",
                options=["Not selected", "Yes", "No"],
                index=0, horizontal=True, key="ca_input",
            )

            mech_vent_input = st.radio(
                "Mechanically ventilated upon ICU admission?",
                options=["Not selected", "Yes", "No"],
                index=0, horizontal=True, key="mv_input",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="input-group-title">🚶 Frailty Assessment</div>', unsafe_allow_html=True)

            cfs_select = st.selectbox(
                "Clinical Frailty Scale (CFS)",
                options=["— Select CFS score —"] + [CFS_OPTIONS[i] for i in range(1,9)],
                index=0, key="cfs_input",
            )

        with col_r:
            st.markdown('<div class="input-group-title">🧪 Laboratory Values</div>', unsafe_allow_html=True)

            inr_val = st.number_input(
                "International Normalised Ratio (INR)",
                min_value=0.5, max_value=20.0, step=0.1, format="%.2f",
                value=None, placeholder="e.g.  1.20",
                key="inr_input",
                help="If INR not available, use 1.0 as default normal value."
            )
            platelet_val = st.number_input(
                "Platelet count  (×10⁹/L)",
                min_value=1, max_value=1000, step=1,
                value=None, placeholder="e.g.  150",
                key="plt_input",
            )
            hct_val = st.number_input(
                "Hematocrit  (%)",
                min_value=1.0, max_value=70.0, step=0.1, format="%.1f",
                value=None, placeholder="e.g.  35.0",
                key="hct_input",
            )
            cl_val = st.number_input(
                "Chloride  (mmol/L)",
                min_value=60, max_value=140, step=1,
                value=None, placeholder="e.g.  102",
                key="cl_input",
            )

            # Live CCI age contribution preview
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="input-group-title">📐 CCI Age Contribution (Auto)</div>', unsafe_allow_html=True)
            if age_val is not None:
                if 65 <= age_val <= 69:   cci_age = 1
                elif 70 <= age_val <= 79: cci_age = 2
                else:                     cci_age = 4
                st.markdown(f"""
                <div style="background:rgba(0,201,177,0.08);border:1px solid rgba(0,201,177,0.25);
                            border-radius:10px;padding:14px 18px;font-size:14px;">
                    Age <span style="color:#00c9b1;font-weight:700;">{age_val} yrs</span> →
                    CCI age contribution:
                    <span style="font-family:'JetBrains Mono',monospace;color:#00c9b1;
                                 font-weight:700;font-size:18px;">+{cci_age} pts</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(143,163,184,0.06);border:1px solid rgba(143,163,184,0.15);
                            border-radius:10px;padding:14px 18px;font-size:13px;color:#8fa3b8;">
                    Enter age above to see automatic CCI age contribution.
                </div>
                """, unsafe_allow_html=True)
                cci_age = 0

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.25);
                    border-radius:10px;padding:12px 18px;font-size:13px;color:#fcd34d;">
            ℹ️ After completing this tab, proceed to <strong>COMORBIDITIES (CCI)</strong> to enter the patient's comorbid conditions,
            then view results in the <strong>RESULTS</strong> tab.
        </div>
        """, unsafe_allow_html=True)

    # ─── TAB 2: CCI ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)

        # Age contribution display
        if "age_input" in st.session_state and st.session_state.age_input is not None:
            a = st.session_state.age_input
            cci_age = 1 if 65<=a<=69 else (2 if 70<=a<=79 else 4)
        else:
            cci_age = 0

        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;">
            <div style="background:rgba(0,201,177,0.08);border:1px solid rgba(0,201,177,0.25);
                        border-radius:10px;padding:14px 22px;flex:1;min-width:200px;">
                <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8fa3b8;margin-bottom:6px;">AGE CONTRIBUTION</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:700;color:#00c9b1;">{cci_age} pts</div>
                <div style="font-size:12px;color:#8fa3b8;">Auto-calculated from age input</div>
            </div>
            <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);
                        border-radius:10px;padding:14px 22px;flex:1;min-width:200px;" id="cci-total-box">
                <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8fa3b8;margin-bottom:6px;">CONDITION POINTS</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:700;color:#a78bfa;" id="cond-pts">—</div>
                <div style="font-size:12px;color:#8fa3b8;">Select conditions below</div>
            </div>
            <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.25);
                        border-radius:10px;padding:14px 22px;flex:1;min-width:200px;">
                <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#8fa3b8;margin-bottom:6px;">TOTAL CCI SCORE</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:700;color:#f43f5e;" id="total-cci">—</div>
                <div style="font-size:12px;color:#8fa3b8;">Will update as you select</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-group-title">🩺 Select All Applicable Comorbid Conditions</div>', unsafe_allow_html=True)
        st.caption("Each condition adds to the CCI score. Tick all that apply to this patient.")

        selected_conditions = []
        # Render in 2 columns
        cond_items = list(CCI_CONDITIONS.items())
        half = len(cond_items)//2 + len(cond_items)%2
        cc1, cc2 = st.columns(2, gap="large")
        for i, (cond, pts) in enumerate(cond_items):
            col = cc1 if i < half else cc2
            with col:
                if st.checkbox(f"{cond}  (+{pts})", key=f"cci_{cond}"):
                    selected_conditions.append((cond, pts))

        cci_cond_pts = sum(p for _,p in selected_conditions)
        cci_total = cci_age + cci_cond_pts

        st.markdown(f"""
        <div style="margin-top:20px;background:rgba(0,201,177,0.06);border:1px solid rgba(0,201,177,0.3);
                    border-radius:12px;padding:16px 22px;display:flex;align-items:center;gap:16px;">
            <div style="font-size:13px;color:#8fa3b8;">Calculated CCI:</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;color:#00c9b1;">{cci_total}</div>
            <div style="font-size:13px;color:#8fa3b8;">
                = Age pts ({cci_age}) + Condition pts ({cci_cond_pts})
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── TAB 3: Results ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)

        # Gather all values from session state safely
        age_v    = st.session_state.get("age_input", None)
        ca_v     = st.session_state.get("ca_input", "Not selected")
        mv_v     = st.session_state.get("mv_input", "Not selected")
        inr_v    = st.session_state.get("inr_input", None)
        plt_v    = st.session_state.get("plt_input", None)
        hct_v    = st.session_state.get("hct_input", None)
        cl_v     = st.session_state.get("cl_input", None)
        cfs_v    = st.session_state.get("cfs_input", "— Select CFS score —")

        # Age CCI re-derive
        if age_v is not None:
            cci_age_r = 1 if 65<=age_v<=69 else (2 if 70<=age_v<=79 else 4)
        else:
            cci_age_r = 0
        cci_cond_r = sum(p for c,p in CCI_CONDITIONS.items()
                         if st.session_state.get(f"cci_{c}", False))
        cci_r = cci_age_r + cci_cond_r

        # Validate
        errors = []
        if age_v is None:                          errors.append("Age")
        if ca_v == "Not selected":                 errors.append("Cardiac arrest status")
        if mv_v == "Not selected":                 errors.append("Mechanical ventilation")
        if inr_v is None:                          errors.append("INR")
        if plt_v is None:                          errors.append("Platelet count")
        if hct_v is None:                          errors.append("Hematocrit")
        if cl_v is None:                           errors.append("Chloride")
        if cfs_v == "— Select CFS score —":        errors.append("Clinical Frailty Scale")

        if errors:
            st.markdown(f"""
            <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.35);
                        border-radius:12px;padding:20px 24px;margin-bottom:20px;">
                <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;
                            color:#f43f5e;margin-bottom:10px;">⚠️ Incomplete Patient Data</div>
                <div style="font-size:13px;color:#8fa3b8;line-height:1.8;">
                    Please complete the following fields in the <strong style="color:#f0f6ff;">Patient Details</strong> tab before viewing results:<br>
                    <strong style="color:#f0f6ff;">{" · ".join(errors)}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Parse
        ca_num  = 1 if ca_v == "Yes" else 0
        mv_num  = 1 if mv_v == "Yes" else 0
        cfs_num = int(cfs_v.split("—")[0].strip())

        # Compute
        composite, exp_s, breakdown = compute(
            age_v, ca_num, inr_v, mv_num, cci_r, cfs_num, hct_v, cl_v
        )
        m7  = get_mort_at(exp_s, 7)
        m14 = get_mort_at(exp_s, 14)
        m30 = get_mort_at(exp_s, 30)
        rl  = classify_risk(m30)
        rc  = RISK_CONFIG[rl]

        # ── Risk Banner ──
        st.markdown(f"""
        <div class="risk-banner" style="background:{rc['bg']};border-color:{rc['border']};color:{rc['color']};">
            <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                <div style="font-size:48px;line-height:1;">
                    {"🟢" if rl=="LOW" else "🟡" if rl=="MODERATE" else "🔴" if rl=="HIGH" else "🟣"}
                </div>
                <div style="flex:1;">
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;
                                letter-spacing:0.1em;text-transform:uppercase;opacity:0.7;margin-bottom:6px;">
                        30-Day Mortality Risk Classification
                    </div>
                    <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
                        <span style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:700;">{rl}</span>
                        <span style="background:{rc['color']};color:#05101f;border-radius:20px;
                                     padding:4px 16px;font-size:13px;font-weight:700;">{rc['band']}</span>
                    </div>
                    <div style="font-size:13px;opacity:0.85;margin-top:8px;line-height:1.6;">
                        <strong>Clinical Recommendation:</strong> {rc['rec']}
                    </div>
                </div>
                <div style="text-align:right;min-width:100px;">
                    <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;
                                text-transform:uppercase;opacity:0.7;">30-Day Risk</div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:48px;
                                font-weight:700;line-height:1;">{m30:.1f}<span style="font-size:24px;">%</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metric Cards ──
        mc1, mc2, mc3, mc4 = st.columns(4, gap="small")
        cards = [
            ("Composite Score",  f"{composite:.4f}", "#00c9b1", "Cox PH model output"),
            ("7-Day Mortality",  f"{m7:.2f}%",       RISK_CONFIG[classify_risk(m7)]["color"],  classify_risk(m7)),
            ("14-Day Mortality", f"{m14:.2f}%",      RISK_CONFIG[classify_risk(m14)]["color"], classify_risk(m14)),
            ("30-Day Mortality", f"{m30:.2f}%",      rc["color"],                              rl),
        ]
        for col, (lbl, val, clr, sub) in zip([mc1, mc2, mc3, mc4], cards):
            with col:
                st.markdown(f"""
                <div class="metric-stat">
                    <div class="label">{lbl}</div>
                    <div class="value" style="color:{clr};font-size:28px;">{val}</div>
                    <div class="sub" style="color:{clr};opacity:0.8;font-weight:600;">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ──
        ch1, ch2 = st.columns([3, 2], gap="large")

        with ch1:
            st.markdown("""
            <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;
                        color:#f0f6ff;margin-bottom:12px;">📈 Mortality Risk Curve — Days 1 to 30</div>
            """, unsafe_allow_html=True)
            days_x = list(range(1, 31))
            mort_y = [get_mort_at(exp_s, d) for d in days_x]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days_x, y=mort_y, mode="lines",
                line=dict(color=rc["color"], width=2.5),
                fill="tozeroy",
                fillcolor=rc["bg"].replace("0.08","0.12"),
                hovertemplate="Day %{x}: %{y:.2f}%<extra></extra>",
            ))
            for kd, kn in [(7,"D7"),(14,"D14"),(30,"D30")]:
                kv = get_mort_at(exp_s,kd)
                fig.add_vline(x=kd, line_dash="dot", line_color="rgba(143,163,184,0.4)", line_width=1)
                fig.add_annotation(x=kd, y=kv+max(mort_y)*0.09,
                                   text=f"<b>{kn}: {kv:.1f}%</b>",
                                   showarrow=False, font=dict(size=11, color=rc["color"]))
            fig.update_layout(
                xaxis=dict(title="ICU Day", color="#8fa3b8", gridcolor="rgba(143,163,184,0.1)", showgrid=True),
                yaxis=dict(title="Mortality Risk (%)", color="#8fa3b8", gridcolor="rgba(143,163,184,0.1)",
                           range=[0, max(max(mort_y)*1.3, 5)]),
                height=300, margin=dict(l=40,r=20,t=10,b=40),
                paper_bgcolor="rgba(10,31,56,0.7)", plot_bgcolor="rgba(5,16,31,0.4)",
                showlegend=False,
                font=dict(family="DM Sans", color="#8fa3b8"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.markdown("""
            <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;
                        color:#f0f6ff;margin-bottom:12px;">🧩 Score Contributions</div>
            """, unsafe_allow_html=True)
            sb_df = pd.DataFrame([
                {"Factor": k, "Value": v, "Color": "#f43f5e" if v >= 0 else "#10b981"}
                for k, v in breakdown.items()
            ]).sort_values("Value", ascending=True)

            fig2 = go.Figure(go.Bar(
                x=sb_df["Value"], y=sb_df["Factor"], orientation="h",
                marker_color=sb_df["Color"],
                text=sb_df["Value"].apply(lambda x: f"{x:+.3f}"),
                textposition="outside",
                textfont=dict(size=10, family="JetBrains Mono"),
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
            ))
            fig2.update_layout(
                xaxis=dict(title="Contribution", color="#8fa3b8", gridcolor="rgba(143,163,184,0.1)"),
                yaxis=dict(color="#8fa3b8"),
                height=300, margin=dict(l=10,r=60,t=10,b=40),
                paper_bgcolor="rgba(10,31,56,0.7)", plot_bgcolor="rgba(5,16,31,0.4)",
                showlegend=False,
                font=dict(family="DM Sans", color="#8fa3b8"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Score Breakdown Table ──
        st.markdown("""
        <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;
                    color:#f0f6ff;margin:8px 0 12px 0;">🔬 Cox PH Score Breakdown</div>
        """, unsafe_allow_html=True)

        beta_list = list(BETA.values())
        rows = [{"Factor": k, "β Coefficient": f"{beta_list[i]:+.5f}", "Contribution": f"{v:+.6f}"}
                for i, (k, v) in enumerate(breakdown.items())]
        rows.append({"Factor": "▶  COMPOSITE SCORE (Σ)", "β Coefficient": "—", "Contribution": f"{composite:+.6f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # ── Full table collapsible ──
        with st.expander("📋 Full Day-by-Day Survival & Mortality Table"):
            trows = []
            for d in sorted(BASELINE_HAZARD.keys()):
                m = get_mort_at(exp_s, d)
                trows.append({"Day": d, "Survival (%)": f"{100-m:.4f}", "Mortality (%)": f"{m:.4f}", "Band": classify_risk(m)})
            st.dataframe(pd.DataFrame(trows), hide_index=True, use_container_width=True)

        # ── New patient ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄  New Patient Assessment", type="secondary", key="new_pt"):
            for k in list(st.session_state.keys()):
                if k not in ["page"]:
                    del st.session_state[k]
            st.session_state.page = "calculator"
            st.rerun()

        # Disclaimer
        st.markdown("""
        <div style="margin-top:24px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);
                    border-radius:10px;padding:14px 18px;font-size:12px;color:#8fa3b8;line-height:1.7;">
            ⚠️ <strong style="color:#fcd34d;">Clinical Disclaimer:</strong>
            This tool is a clinical decision-support aid for trained professionals only. It does not replace
            multidisciplinary team judgement or individualised patient care. All estimates are probabilistic
            outputs of a validated Cox Proportional Hazard model.
            <strong>Version 1.0 — September 2025.</strong>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "about":
    col_back, _ = st.columns([1, 6])
    with col_back:
        if st.button("← Back", key="back_about"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;
                color:#f0f6ff;margin-bottom:6px;">About This Tool</div>
    <div style="font-size:14px;color:#8fa3b8;margin-bottom:28px;">
        Methodology, model variables, and clinical interpretation guide.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="card">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;
                        color:#00c9b1;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:14px;">
                ⚙️ Model Overview
            </div>
            <p style="font-size:14px;color:#8fa3b8;line-height:1.8;">
                This calculator implements a <strong style="color:#f0f6ff;">Cox Proportional Hazard (CPH) model</strong>
                validated for elderly ICU patients (≥ 65 years). The model estimates the probability of
                in-ICU mortality at specified time points using a composite risk score derived from
                clinical, laboratory, and functional parameters.
            </p>
            <p style="font-size:14px;color:#8fa3b8;line-height:1.8;margin-top:12px;">
                The survival function is:
            </p>
            <div style="background:rgba(0,201,177,0.06);border:1px solid rgba(0,201,177,0.2);
                        border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace;
                        font-size:13px;color:#00c9b1;margin:10px 0;">
                S(t) = exp(−H₀(t) × exp(Σ βᵢXᵢ))
            </div>
            <p style="font-size:13px;color:#8fa3b8;line-height:1.7;">
                Where H₀(t) is the baseline cumulative hazard at day t, βᵢ are the model
                coefficients, and Xᵢ are the patient's clinical values.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;
                        color:#00c9b1;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:14px;">
                📐 Model Variables & Coefficients
            </div>
        """, unsafe_allow_html=True)

        model_info = [
            ("Cardiac Arrest", "Binary (0/1)", "+2.7268", "Strongest predictor"),
            ("Loge INR", "ln(INR)", "+1.7000", "Coagulopathy marker"),
            ("Mechanical Ventilation", "Binary (0/1)", "+0.9076", "Respiratory failure"),
            ("Loge Age", "ln(Age)", "+1.5523", "Age-adjusted risk"),
            ("Loge CCI", "ln(CCI)", "+0.8597", "Comorbidity burden"),
            ("CFS Score", "1–8 scale", "+0.0598", "Frailty assessment"),
            ("Hematocrit", "%", "−0.0583", "Anemia / perfusion"),
            ("Chloride", "mmol/L", "−0.00096", "Electrolyte balance"),
        ]
        for var, typ, beta, note in model_info:
            color = "#f43f5e" if "+" in beta else "#10b981"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:8px 0;
                        border-bottom:1px solid rgba(0,201,177,0.08);">
                <div style="flex:2;font-size:13px;color:#f0f6ff;font-weight:500;">{var}</div>
                <div style="flex:1;font-size:11px;color:#8fa3b8;">{typ}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                            color:{color};font-weight:600;min-width:70px;">{beta}</div>
                <div style="flex:2;font-size:11px;color:#8fa3b8;">{note}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;
                    color:#00c9b1;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">
            🏥 Risk Band Clinical Guidance
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
    """, unsafe_allow_html=True)

    for rl, cfg in RISK_CONFIG.items():
        st.markdown(f"""
        <div style="background:{cfg['bg']};border:1px solid {cfg['border']};border-radius:10px;padding:16px;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;
                        color:{cfg['color']};margin-bottom:4px;">{rl}</div>
            <div style="font-size:22px;font-weight:700;color:{cfg['color']};margin-bottom:8px;">{cfg['band']}</div>
            <div style="font-size:12px;color:#8fa3b8;line-height:1.6;">{cfg['rec']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬  Start Patient Assessment", type="primary"):
        st.session_state.page = "calculator"
        st.rerun()
