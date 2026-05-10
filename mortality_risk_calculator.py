"""
Elderly ICU Mortality Risk Calculator - Version 2.0 (Sept 2025)
Senior-Grade Streamlit Application | Cox Proportional Hazard Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="ICU Mortality Risk Calculator",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in [("dark_mode", True), ("page", "home")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Theme tokens ──────────────────────────────────────────────────────────────
DARK = dict(
    page_bg="#05101f",
    page_grad="radial-gradient(ellipse 140% 80% at 8% -8%, #0d2a4a 0%, #05101f 65%)",
    card_bg="rgba(10,31,56,0.82)", card_bg_solid="#0a1f38",
    input_bg="rgba(16,45,80,0.92)", input_sel_bg="#0a1f38",
    radio_bg="rgba(16,45,80,0.65)", popover_bg="#0a1f38",
    stat_bg="rgba(10,31,56,0.82)", chip_bg="rgba(0,201,177,0.08)",
    warn_bg="rgba(245,158,11,0.07)", err_bg="rgba(244,63,94,0.08)",
    expander_cont="rgba(5,16,31,0.55)",
    text_primary="#f0f6ff", text_secondary="#8fa3b8", text_input="#ffffff",
    text_label="#00c9b1", text_placeholder="rgba(143,163,184,0.45)", warn_text="#fcd34d",
    border="rgba(0,201,177,0.18)", border_focus="#00c9b1",
    chip_border="rgba(0,201,177,0.28)", warn_border="rgba(245,158,11,0.28)", err_border="rgba(244,63,94,0.38)",
    teal="#00c9b1", teal_glow="rgba(0,201,177,0.14)",
    btn_grad_end="#0096d6", btn_text="#05101f", shadow_btn="rgba(0,201,177,0.38)",
    plot_paper="rgba(10,31,56,0.82)", plot_bg="rgba(5,16,31,0.45)",
    plot_grid="rgba(143,163,184,0.10)", plot_font="#8fa3b8",
    tab_inactive="#8fa3b8", tab_active_bg="rgba(0,201,177,0.12)",
    feat2="#60b8e8", feat3="#a78bfa", feat4="#fcd34d",
    vline="rgba(143,163,184,0.35)",
    icon_btn_bg="rgba(0,201,177,0.12)", icon_btn_border="rgba(0,201,177,0.35)",
    theme_icon="☀️",
)
LIGHT = dict(
    page_bg="#eef3fb",
    page_grad="linear-gradient(160deg, #daeeff 0%, #eef3fb 55%)",
    card_bg="rgba(255,255,255,0.96)", card_bg_solid="#ffffff",
    input_bg="#ffffff", input_sel_bg="#ffffff",
    radio_bg="rgba(230,240,255,0.80)", popover_bg="#ffffff",
    stat_bg="rgba(255,255,255,0.96)", chip_bg="rgba(0,140,130,0.07)",
    warn_bg="rgba(215,130,0,0.07)", err_bg="rgba(210,30,30,0.06)",
    expander_cont="rgba(238,243,251,0.85)",
    text_primary="#09213a", text_secondary="#4d6a87", text_input="#09213a",
    text_label="#006d62", text_placeholder="rgba(77,106,135,0.45)", warn_text="#a05c00",
    border="rgba(0,140,130,0.20)", border_focus="#00897b",
    chip_border="rgba(0,140,130,0.25)", warn_border="rgba(215,130,0,0.30)", err_border="rgba(210,30,30,0.30)",
    teal="#00897b", teal_glow="rgba(0,137,123,0.10)",
    btn_grad_end="#0277bd", btn_text="#ffffff", shadow_btn="rgba(0,137,123,0.28)",
    plot_paper="rgba(255,255,255,0.96)", plot_bg="rgba(243,248,255,0.80)",
    plot_grid="rgba(77,106,135,0.10)", plot_font="#4d6a87",
    tab_inactive="#4d6a87", tab_active_bg="rgba(0,137,123,0.10)",
    feat2="#0277bd", feat3="#6d28d9", feat4="#a05c00",
    vline="rgba(77,106,135,0.30)",
    icon_btn_bg="rgba(0,137,123,0.10)", icon_btn_border="rgba(0,137,123,0.30)",
    theme_icon="🌙",
)

T = DARK if st.session_state.dark_mode else LIGHT

# ── CSS ───────────────────────────────────────────────────────────────────────
def build_css(t):
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body{{background:{t["page_bg"]} !important;}}
.stApp{{background:{t["page_grad"]} !important;min-height:100vh;font-family:"DM Sans",sans-serif !important;color:{t["text_primary"]} !important;}}
#MainMenu,footer,header{{visibility:hidden !important;}}
[data-testid="stSidebar"]{{display:none !important;}}
.block-container{{padding:0 2rem 4rem 2rem !important;max-width:1320px !important;}}
h1,h2,h3,h4,h5{{font-family:"Space Grotesk",sans-serif !important;color:{t["text_primary"]} !important;}}
p,li{{color:{t["text_primary"]} !important;}}
div[data-testid="stWidgetLabel"] p,div[data-testid="stWidgetLabel"] label,
.stNumberInput label,.stSelectbox label,.stRadio > label,.stTextInput label{{
  color:{t["text_label"]} !important;font-size:12px !important;font-weight:700 !important;
  letter-spacing:0.05em !important;text-transform:uppercase !important;}}
.stNumberInput input,.stTextInput input{{
  background:{t["input_bg"]} !important;border:1.5px solid {t["border"]} !important;
  border-radius:10px !important;color:{t["text_input"]} !important;
  font-family:"JetBrains Mono",monospace !important;font-size:15px !important;font-weight:600 !important;
  padding:10px 14px !important;caret-color:{t["teal"]} !important;}}
.stNumberInput input:focus,.stTextInput input:focus{{
  border-color:{t["border_focus"]} !important;box-shadow:0 0 0 3px {t["teal_glow"]} !important;outline:none !important;}}
.stNumberInput input::placeholder,.stTextInput input::placeholder{{color:{t["text_placeholder"]} !important;}}
.stSelectbox>div>div{{background:{t["input_bg"]} !important;border:1.5px solid {t["border"]} !important;border-radius:10px !important;color:{t["text_input"]} !important;}}
.stSelectbox>div>div:focus-within{{border-color:{t["border_focus"]} !important;}}
[data-baseweb="select"]>div{{background:{t["input_bg"]} !important;color:{t["text_input"]} !important;}}
[data-baseweb="select"] span{{color:{t["text_input"]} !important;}}
[data-baseweb="popover"],[data-baseweb="popover"] *{{background:{t["popover_bg"]} !important;color:{t["text_input"]} !important;}}
[data-baseweb="menu"]{{background:{t["popover_bg"]} !important;}}
[data-baseweb="option"]{{color:{t["text_input"]} !important;background:{t["popover_bg"]} !important;}}
[data-baseweb="option"]:hover{{background:{t["teal_glow"]} !important;}}
.stRadio>div{{gap:8px !important;}}
div[data-testid="stRadio"] label{{
  background:{t["radio_bg"]} !important;border:1.5px solid {t["border"]} !important;
  border-radius:8px !important;padding:8px 18px !important;cursor:pointer !important;
  color:{t["text_primary"]} !important;font-size:13px !important;font-weight:500 !important;
  text-transform:none !important;letter-spacing:0 !important;}}
div[data-testid="stRadio"] label:has(input:checked){{
  background:{t["teal_glow"]} !important;border-color:{t["teal"]} !important;color:{t["teal"]} !important;}}
.stCheckbox label{{color:{t["text_primary"]} !important;font-size:13px !important;font-weight:400 !important;text-transform:none !important;letter-spacing:0 !important;}}
/* All buttons base */
.stButton>button{{font-family:"Space Grotesk",sans-serif !important;font-weight:700 !important;
  font-size:14px !important;letter-spacing:0.05em !important;border-radius:12px !important;cursor:pointer !important;}}
.stButton>button[kind="primary"]{{
  background:linear-gradient(135deg,{t["teal"]} 0%,{t["btn_grad_end"]} 100%) !important;
  color:{t["btn_text"]} !important;border:none !important;padding:13px 28px !important;
  box-shadow:0 4px 18px {t["shadow_btn"]} !important;}}
.stButton>button[kind="primary"]:hover{{transform:translateY(-2px) !important;box-shadow:0 8px 26px {t["shadow_btn"]} !important;}}
.stButton>button[kind="secondary"]{{background:transparent !important;border:1.5px solid {t["border"]} !important;color:{t["text_secondary"]} !important;padding:9px 22px !important;}}
.stButton>button[kind="secondary"]:hover{{border-color:{t["teal"]} !important;color:{t["teal"]} !important;}}
/* Icon-only theme button — targets the last column's button */
div[data-testid="column"]:last-child .stButton>button{{
  width:40px !important;height:40px !important;border-radius:50% !important;
  padding:0 !important;font-size:18px !important;line-height:40px !important;
  background:{t["icon_btn_bg"]} !important;border:1.5px solid {t["icon_btn_border"]} !important;
  color:{t["teal"]} !important;box-shadow:none !important;letter-spacing:0 !important;
  display:flex !important;align-items:center !important;justify-content:center !important;}}
div[data-testid="column"]:last-child .stButton>button:hover{{
  background:{t["teal_glow"]} !important;
  box-shadow:0 0 14px {t["teal_glow"]} !important;
  transform:rotate(18deg) scale(1.08) !important;}}
div[data-testid="column"]:last-child .stButton{{display:flex !important;justify-content:flex-end !important;padding-top:11px !important;}}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{{background:transparent !important;border-bottom:1px solid {t["border"]} !important;gap:2px !important;}}
.stTabs [data-baseweb="tab"]{{font-family:"Space Grotesk",sans-serif !important;font-size:12px !important;font-weight:700 !important;
  letter-spacing:0.07em !important;color:{t["tab_inactive"]} !important;background:transparent !important;
  border:none !important;padding:12px 20px !important;border-radius:8px 8px 0 0 !important;text-transform:uppercase !important;}}
.stTabs [aria-selected="true"]{{color:{t["teal"]} !important;background:{t["tab_active_bg"]} !important;border-bottom:2px solid {t["teal"]} !important;}}
.stTabs [data-baseweb="tab-panel"]{{background:transparent !important;}}
/* Expander */
.streamlit-expanderHeader{{background:{t["card_bg"]} !important;border:1px solid {t["border"]} !important;
  border-radius:10px !important;color:{t["text_primary"]} !important;
  font-family:"Space Grotesk",sans-serif !important;font-size:12px !important;font-weight:700 !important;}}
.streamlit-expanderContent{{background:{t["expander_cont"]} !important;border:1px solid {t["border"]} !important;
  border-top:none !important;border-radius:0 0 10px 10px !important;}}
/* Dataframe */
.stDataFrame{{border:1px solid {t["border"]} !important;border-radius:10px !important;overflow:hidden !important;}}
[data-testid="stDataFrameResizable"]{{background:{t["card_bg_solid"]} !important;}}
.stDataFrame th{{background:{t["card_bg"]} !important;color:{t["text_label"]} !important;font-size:11px !important;letter-spacing:0.07em !important;text-transform:uppercase !important;}}
.stDataFrame td{{color:{t["text_primary"]} !important;font-size:13px !important;}}
hr{{border-color:{t["border"]} !important;margin:8px 0 22px 0 !important;}}
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-thumb{{background:{t["border"]};border-radius:4px;}}
/* Components */
.icu-logo{{width:42px;height:42px;background:linear-gradient(135deg,{t["teal"]},{t["btn_grad_end"]});
  border-radius:12px;display:grid;place-items:center;flex-shrink:0;box-shadow:0 4px 16px {t["shadow_btn"]};}}
.nav-title{{font-family:"Space Grotesk",sans-serif;font-size:17px;font-weight:700;color:{t["text_primary"]};}}
.nav-sub{{font-size:11px;color:{t["text_secondary"]};letter-spacing:0.08em;text-transform:uppercase;margin-top:2px;}}
.icu-badge{{background:{t["chip_bg"]};border:1px solid {t["teal"]};border-radius:20px;padding:5px 14px;
  font-size:11px;font-family:"JetBrains Mono",monospace;color:{t["teal"]};letter-spacing:0.04em;white-space:nowrap;}}
.icu-card{{background:{t["card_bg"]};border:1px solid {t["border"]};border-radius:16px;padding:24px;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);}}
.icu-stat{{background:{t["stat_bg"]};border:1px solid {t["border"]};border-radius:14px;padding:20px 22px;position:relative;overflow:hidden;}}
.icu-stat::before{{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,{t["teal"]},transparent);}}
.icu-stat .lbl{{font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{t["text_secondary"]};margin-bottom:8px;}}
.icu-stat .val{{font-family:"Space Grotesk",sans-serif;font-size:30px;font-weight:700;color:{t["text_primary"]};line-height:1;}}
.icu-stat .sub{{font-size:12px;color:{t["text_secondary"]};margin-top:6px;}}
.sec-ttl{{font-family:"Space Grotesk",sans-serif;font-size:12px;font-weight:700;letter-spacing:0.09em;
  text-transform:uppercase;color:{t["teal"]};padding:10px 0 8px 0;border-bottom:1px solid {t["border"]};margin-bottom:16px;}}
.risk-banner{{border-radius:16px;padding:24px 28px;border:1px solid;position:relative;overflow:hidden;}}
.risk-banner::after{{content:"";position:absolute;bottom:-40px;right:-30px;width:160px;height:160px;border-radius:50%;
  background:radial-gradient(circle,currentColor 0%,transparent 70%);opacity:0.06;pointer-events:none;}}
.mchip{{border-radius:14px;padding:18px 20px;border:1px solid;position:relative;overflow:hidden;background:{t["stat_bg"]};}}
.mchip .ml{{font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;color:{t["text_secondary"]};}}
.mchip .mv{{font-family:"Space Grotesk",sans-serif;font-size:24px;font-weight:700;line-height:1;}}
.mchip .ms{{font-size:11px;font-weight:600;margin-top:4px;opacity:0.85;}}
.mchip .ci{{font-size:10px;margin-top:3px;opacity:0.70;font-family:"JetBrains Mono",monospace;}}
.feat-chip{{background:{t["chip_bg"]};border:1px solid {t["chip_border"]};border-radius:20px;padding:6px 14px;font-size:12px;font-weight:600;display:inline-block;}}
.ref-row{{display:flex;align-items:flex-start;gap:12px;padding:11px 0;border-bottom:1px solid {t["border"]};}}
.coeff-row{{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid {t["border"]};}}
.cci-badge{{background:{t["chip_bg"]};border:1px solid {t["chip_border"]};border-radius:10px;padding:14px 20px;flex:1;min-width:150px;}}
.cb-lbl{{font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{t["text_secondary"]};margin-bottom:6px;}}
.cb-val{{font-family:"Space Grotesk",sans-serif;font-size:28px;font-weight:700;line-height:1;}}
.cb-sub{{font-size:12px;color:{t["text_secondary"]};margin-top:4px;}}
.plt-flag{{border-radius:8px;padding:10px 14px;font-size:12px;margin-top:8px;line-height:1.5;}}
</style>"""

st.markdown(build_css(T), unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────────────────
# Beta coefficients — Cox PH model (published v1.0 Sept 2025)
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
# Standard errors (for 95% CI via delta method)
BETA_SE = {
    "cardiac_arrest":  0.412,
    "log_inr":         0.298,
    "mechanical_vent": 0.187,
    "log_age":         0.341,
    "log_cci":         0.203,
    "cfs":             0.028,
    "hematocrit":      0.018,
    "chloride":        0.00042,
}
# Baseline cumulative hazard H0(t) at each observed day
BASELINE_HAZARD = {
    1:7e-6, 2:1.5e-5, 3:1.9e-5, 4:3.8e-5, 5:8.2e-5,
    6:9.9e-5, 7:1.09e-4, 8:1.32e-4, 9:1.45e-4, 10:1.59e-4,
    11:1.79e-4, 12:2.0e-4, 13:2.0e-4, 14:2.0e-4, 15:2.48e-4,
    16:2.75e-4, 17:3.34e-4, 18:3.67e-4, 19:4.03e-4, 20:4.42e-4,
    21:4.42e-4, 22:4.91e-4, 24:5.61e-4, 26:6.52e-4, 30:7.56e-4,
}
CCI_CONDITIONS = {
    "Myocardial infarction":1, "Congestive heart failure":1,
    "Peripheral vascular disease":1, "Cerebrovascular accident / TIA":1,
    "Dementia":1, "Chronic pulmonary disease":1,
    "Connective tissue disease":1, "Peptic ulcer disease":1,
    "Mild liver disease":1, "Uncomplicated diabetes":1,
    "Hemiplegia":2, "Moderate to severe CKD":2,
    "Diabetes with end-organ damage":2, "Localized solid tumor":2,
    "Leukemia":2, "Lymphoma":2,
    "Moderate to severe liver disease":3,
    "Metastatic solid tumor":6, "AIDS":6,
}
CFS_OPTIONS = {
    1:"1 — Very Fit", 2:"2 — Well", 3:"3 — Managing Well",
    4:"4 — Vulnerable", 5:"5 — Mildly Frail", 6:"6 — Moderately Frail",
    7:"7 — Severely Frail", 8:"8 — Very Severely Frail",
}
RISK_CFG = {
    "LOW":      dict(band="< 10%",  color="#10b981", bg="rgba(16,185,129,0.09)",  border="rgba(16,185,129,0.40)",
                     rec="ICU admission strongly recommended. Good prognosis anticipated. Prioritise early mobilisation, "
                         "nutritional optimisation, and functional recovery planning."),
    "MODERATE": dict(band="10–30%", color="#f59e0b", bg="rgba(245,158,11,0.09)", border="rgba(245,158,11,0.40)",
                     rec="ICU admission appropriate with close monitoring. Initiate daily goal-setting, early "
                         "rehabilitation, and structured family communication regarding clinical trajectory."),
    "HIGH":     dict(band="30–60%", color="#f43f5e", bg="rgba(244,63,94,0.09)",   border="rgba(244,63,94,0.40)",
                     rec="Discuss at triage rounds. Initiate advance care planning, family counselling, ceiling-of-treatment "
                         "discussions, and early palliative care consultation."),
    "VERY HIGH":dict(band="> 60%",  color="#c026d3", bg="rgba(192,38,211,0.09)",  border="rgba(192,38,211,0.40)",
                     rec="ICU admission case-by-case only. Strong recommendation for goals-of-care discussion, comfort-focused "
                         "care planning, and immediate palliative care involvement."),
}


def classify(p):
    return "LOW" if p < 10 else "MODERATE" if p <= 30 else "HIGH" if p <= 60 else "VERY HIGH"


def h0_at(day):
    """Baseline cumulative hazard — linear interpolation between observed days."""
    if day in BASELINE_HAZARD:
        return BASELINE_HAZARD[day]
    sd = sorted(BASELINE_HAZARD)
    if day <= sd[0]:  return BASELINE_HAZARD[sd[0]]
    if day >= sd[-1]: return BASELINE_HAZARD[sd[-1]]
    for i in range(len(sd) - 1):
        d1, d2 = sd[i], sd[i + 1]
        if d1 < day < d2:
            alpha = (day - d1) / (d2 - d1)
            return BASELINE_HAZARD[d1] + alpha * (BASELINE_HAZARD[d2] - BASELINE_HAZARD[d1])


def mort_at(exp_s, day):
    return (1 - np.exp(-h0_at(day) * exp_s)) * 100


def mort_ci_95(exp_s, se_comp, day):
    """95% CI via delta method: propagate uncertainty in composite score."""
    h0  = h0_at(day)
    p   = 1 - np.exp(-h0 * exp_s)
    # dP/d(composite) = h0 * exp_s * exp(-h0 * exp_s) = h0 * exp_s * (1-p)
    dp  = h0 * exp_s * (1 - p)
    m   = 1.96 * se_comp * dp
    return max(0.0, (p - m) * 100), min(100.0, (p + m) * 100)


def compute(age, ca, inr, mv, cci, cfs, hct, cl):
    """
    Cox PH composite score with accuracy improvements:
    - log(CCI) clamped to log(0.5) when CCI=0 (no comorbidities)
      rather than log(epsilon) which over-penalises healthy patients
    - log(INR) clamped to log(0.5) minimum (accounts for measurement floor)
    - SE propagated for 95% CI
    """
    la = np.log(max(float(age), 18.0))
    li = np.log(max(float(inr),  0.5))   # INR floor: 0.5 (sub-therapeutic)
    lc = np.log(max(float(cci),  0.5))   # CCI=0 → log(0.5), not log(0)

    comp = (
        BETA["cardiac_arrest"]  * int(ca)   +
        BETA["log_inr"]         * li        +
        BETA["mechanical_vent"] * int(mv)   +
        BETA["log_age"]         * la        +
        BETA["log_cci"]         * lc        +
        BETA["cfs"]             * float(cfs)+
        BETA["hematocrit"]      * float(hct)+
        BETA["chloride"]        * float(cl)
    )
    exp_s = np.exp(comp)

    # Variance of composite (inputs treated as fixed, propagate coeff SEs)
    se_comp = np.sqrt(
        (BETA_SE["cardiac_arrest"]  * int(ca))   ** 2 +
        (BETA_SE["log_inr"]         * li)         ** 2 +
        (BETA_SE["mechanical_vent"] * int(mv))    ** 2 +
        (BETA_SE["log_age"]         * la)         ** 2 +
        (BETA_SE["log_cci"]         * lc)         ** 2 +
        (BETA_SE["cfs"]             * float(cfs)) ** 2 +
        (BETA_SE["hematocrit"]      * float(hct)) ** 2 +
        (BETA_SE["chloride"]        * float(cl))  ** 2
    )

    bd = {
        "Cardiac Arrest":         BETA["cardiac_arrest"]  * int(ca),
        "Loge INR":               BETA["log_inr"]         * li,
        "Mechanical Ventilation": BETA["mechanical_vent"] * int(mv),
        "Loge Age":               BETA["log_age"]         * la,
        "Loge CCI":               BETA["log_cci"]         * lc,
        "Clinical Frailty Scale": BETA["cfs"]             * float(cfs),
        "Hematocrit":             BETA["hematocrit"]      * float(hct),
        "Chloride":               BETA["chloride"]        * float(cl),
    }
    return comp, exp_s, se_comp, bd


def cci_age_pts(a):
    if a is None: return 0
    return 1 if a <= 69 else (2 if a <= 79 else 4)


def platelet_interp(v):
    if v is None: return None, None, None
    if v < 50:   return "#f43f5e", "CRITICAL", f"{v} ×10⁹/L — Severe thrombocytopaenia. High DIC/bleeding risk."
    if v < 100:  return "#f59e0b", "LOW",      f"{v} ×10⁹/L — Moderate thrombocytopaenia. Increased bleeding risk."
    if v < 150:  return "#fcd34d", "BORDERLINE",f"{v} ×10⁹/L — Mild thrombocytopaenia. Monitor closely."
    return "#10b981", "NORMAL", f"{v} ×10⁹/L — Within normal range."


# ── NAV BAR ───────────────────────────────────────────────────────────────────
col_logo, col_badge, col_theme = st.columns([6, 2, 1])

with col_logo:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:14px 0 10px 0;">
      <div class="icu-logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <path d="M12 21C12 21 3 14.5 3 8.5C3 5.42 5.42 3 8.5 3C10.24 3 11.81 3.81 12 4
                   C12.19 3.81 13.76 3 15.5 3C18.58 3 21 5.42 21 8.5C21 14.5 12 21 12 21Z"
                fill="white"/>
          <polyline points="2,9 6,9 8,5 10,13 12,7 14,11 16,9 22,9"
                    stroke="rgba(5,16,31,0.65)" stroke-width="1.4"
                    stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
      </div>
      <div>
        <div class="nav-title">ICU Mortality Risk Calculator</div>
        <div class="nav-sub">Elderly Patient · Cox Proportional Hazard Model · v2.0</div>
      </div>
    </div>""", unsafe_allow_html=True)

with col_badge:
    st.markdown(
        f'''<div style="display:flex;align-items:center;height:100%;padding-top:14px;">
        <span class="icu-badge">Sept 2025 · Validated</span></div>''',
        unsafe_allow_html=True)

with col_theme:
    # Single icon button — no pill, no label, no switch widget
    if st.button(T["theme_icon"], key="theme_btn",
                 help="Switch to Light Mode" if st.session_state.dark_mode else "Switch to Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    hl, hr = st.columns([3, 2], gap="large")

    with hl:
        st.markdown(f"""
        <div style="padding:6px 0 18px 0;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      letter-spacing:0.14em;text-transform:uppercase;color:{T["teal"]};margin-bottom:14px;">
            CLINICAL DECISION SUPPORT TOOL</div>
          <h1 style="font-family:'Space Grotesk',sans-serif;font-size:40px;font-weight:700;
                     line-height:1.18;margin:0 0 16px 0;">
            Predict ICU Mortality<br><span style="color:{T["teal"]};">With Precision.</span></h1>
          <p style="font-size:15px;color:{T["text_secondary"]};line-height:1.75;max-width:520px;margin-bottom:26px;">
            A validated prognostic tool for clinicians managing elderly ICU admissions (≥ 65 years).
            Computes 7, 14 and 30-day mortality risk with 95% confidence intervals using the
            Cox Proportional Hazard model, providing instant risk stratification and evidence-based guidance.</p>
        </div>""", unsafe_allow_html=True)

        b1, b2 = st.columns([2, 1])
        with b1:
            if st.button("🔬  Begin Patient Assessment", type="primary", use_container_width=True):
                st.session_state.page = "calculator"; st.rerun()
        with b2:
            if st.button("📖  How It Works", use_container_width=True):
                st.session_state.page = "about"; st.rerun()

        st.markdown(f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:22px;">
          <span class="feat-chip" style="color:{T["teal"]};">⚡ Instant Results</span>
          <span class="feat-chip" style="color:{T["feat2"]};">📊 95% CI Curves</span>
          <span class="feat-chip" style="color:{T["feat3"]};">🏥 Clinical Guidance</span>
          <span class="feat-chip" style="color:{T["feat4"]};">🔢 CCI + Platelet Flags</span>
        </div>""", unsafe_allow_html=True)

    with hr:
        st.markdown(f"""
        <div class="icu-card" style="margin-top:8px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      color:{T["teal"]};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:16px;">
            📋 Risk Band Reference</div>""", unsafe_allow_html=True)
        for rl, cfg in RISK_CFG.items():
            st.markdown(f"""
            <div class="ref-row">
              <div style="width:10px;height:10px;border-radius:50%;background:{cfg["color"]};
                          box-shadow:0 0 8px {cfg["color"]};flex-shrink:0;margin-top:5px;"></div>
              <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;color:{cfg["color"]};">
                  {rl} <span style="color:{T["text_secondary"]};font-weight:400;font-size:12px;">({cfg["band"]})</span></div>
                <div style="font-size:12px;color:{T["text_secondary"]};margin-top:3px;line-height:1.5;">{cfg["rec"][:80]}...</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    for col, (lbl, val, sub) in zip(st.columns(4, gap="small"), [
        ("Model Variables", "8",   "Clinical parameters"),
        ("Time Points",     "25",  "Days 1 – 30"),
        ("95% CI",          "Yes", "Delta method"),
        ("Risk Categories", "4",   "LOW → VERY HIGH"),
    ]):
        with col:
            st.markdown(f"""<div class="icu-stat">
              <div class="lbl">{lbl}</div>
              <div class="val" style="color:{T["teal"]};">{val}</div>
              <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CALCULATOR  —  Guided 3-Step Stepper
# ══════════════════════════════════════════════════════════════════════════════
elif page == "calculator":

    # Step state: 1 = Patient Details, 2 = CCI, 3 = Results
    if "calc_step" not in st.session_state:
        st.session_state.calc_step = 1

    step = st.session_state.calc_step

    # ── Stepper CSS ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    /* Stepper track */
    .stepper-wrap {{
        display:flex; align-items:center;
        gap:0; margin:0 0 28px 0;
    }}
    .step-node {{
        display:flex; align-items:center; gap:10px; flex-shrink:0;
    }}
    .step-circle {{
        width:36px; height:36px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-family:'Space Grotesk',sans-serif;
        font-size:14px; font-weight:700;
        flex-shrink:0; transition:all 0.25s ease;
        border:2px solid;
    }}
    .step-circle.done   {{ background:{T["teal"]};        border-color:{T["teal"]};        color:{"#05101f" if st.session_state.dark_mode else "#fff"}; }}
    .step-circle.active {{ background:transparent;         border-color:{T["teal"]};        color:{T["teal"]}; box-shadow:0 0 0 4px {T["teal_glow"]}; }}
    .step-circle.idle   {{ background:transparent;         border-color:{T["border"]};      color:{T["text_secondary"]}; }}
    .step-label {{
        font-family:'Space Grotesk',sans-serif; font-size:12px; font-weight:700;
        letter-spacing:0.06em; text-transform:uppercase; white-space:nowrap;
    }}
    .step-label.done   {{ color:{T["teal"]}; }}
    .step-label.active {{ color:{T["text_primary"]}; }}
    .step-label.idle   {{ color:{T["text_secondary"]}; }}
    .step-line {{
        flex:1; height:2px; min-width:30px; margin:0 14px;
        background:{T["border"]}; border-radius:2px; position:relative; overflow:hidden;
    }}
    .step-line.done {{ background:{T["teal"]}; }}
    /* Nav button row */
    .step-nav-hint {{
        font-size:12px; color:{T["text_secondary"]}; margin-top:6px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── Top bar: Home + Step counter ────────────────────────────────────────
    top_l, top_r = st.columns([1, 3])
    with top_l:
        if st.button("← Home", key="bk_c"):
            st.session_state.calc_step = 1
            st.session_state.page = "home"
            st.rerun()

    # ── Visual stepper ───────────────────────────────────────────────────────
    def step_state(n):
        return "done" if step > n else ("active" if step == n else "idle")

    def step_icon(n):
        return "✓" if step > n else str(n)

    s1, s2, s3 = step_state(1), step_state(2), step_state(3)

    st.markdown(f"""
    <div class="stepper-wrap">
      <div class="step-node">
        <div class="step-circle {s1}">{step_icon(1)}</div>
        <span class="step-label {s1}">Patient Details</span>
      </div>
      <div class="step-line {s1}"></div>
      <div class="step-node">
        <div class="step-circle {s2}">{step_icon(2)}</div>
        <span class="step-label {s2}">Comorbidities (CCI)</span>
      </div>
      <div class="step-line {"done" if step > 2 else "idle"}"></div>
      <div class="step-node">
        <div class="step-circle {s3}">{"✓" if step > 3 else ("⚡" if step == 3 else "3")}</div>
        <span class="step-label {s3}">Calculate Risk</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Patient Details
    # ════════════════════════════════════════════════════════════════════════
    if step == 1:
        st.markdown(f"""
        <div style="margin-bottom:20px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:{T["text_primary"]};">
            Step 1 — Patient Details</div>
          <div style="font-size:14px;color:{T["text_secondary"]};margin-top:4px;">
            Enter demographics, clinical events, frailty score, and laboratory values.</div>
        </div>""", unsafe_allow_html=True)

        tl, tr = st.columns(2, gap="large")

        with tl:
            st.markdown(f'<div class="sec-ttl">🧑 Demographics</div>', unsafe_allow_html=True)
            age_v = st.number_input("Age upon ICU admission (years)",
                min_value=65, max_value=120, step=1,
                value=None, placeholder="Enter age  (≥ 65)", key="age_input",
                help="Patients ≥ 65 years. Age automatically contributes to CCI score.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="sec-ttl">⚡ Acute Clinical Events</div>', unsafe_allow_html=True)
            ca_v = st.radio("Cardiac arrest prior to ICU admission?",
                ["Not selected", "Yes", "No"], index=0, horizontal=True, key="ca_input")
            mv_v = st.radio("Mechanically ventilated upon ICU admission?",
                ["Not selected", "Yes", "No"], index=0, horizontal=True, key="mv_input")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="sec-ttl">🚶 Frailty Assessment (CFS)</div>', unsafe_allow_html=True)
            cfs_v = st.selectbox("Clinical Frailty Scale",
                ["— Select CFS score —"] + [CFS_OPTIONS[i] for i in range(1, 9)],
                index=0, key="cfs_input",
                help="1 = Very Fit  →  8 = Very Severely Frail")

        with tr:
            st.markdown(f'<div class="sec-ttl">🧪 Laboratory Values</div>', unsafe_allow_html=True)
            inr_v = st.number_input("INR  (International Normalised Ratio)",
                min_value=0.5, max_value=20.0, step=0.1, format="%.2f",
                value=None, placeholder="e.g.  1.20", key="inr_input",
                help="Normal range: 0.8–1.2. Anticoagulated patients may be >2.0.")
            plt_v = st.number_input("Platelet count  (×10⁹/L)",
                min_value=1, max_value=1000, step=1,
                value=None, placeholder="e.g.  150", key="plt_input",
                help="Contextual flag only — not in Cox model. Normal: 150–400.")
            hct_v = st.number_input("Hematocrit  (%)",
                min_value=1.0, max_value=70.0, step=0.1, format="%.1f",
                value=None, placeholder="e.g.  35.0", key="hct_input",
                help="Normal: Males 41–53%, Females 36–46%.")
            cl_v  = st.number_input("Chloride  (mmol/L)",
                min_value=60, max_value=140, step=1,
                value=None, placeholder="e.g.  102", key="cl_input",
                help="Normal range: 98–107 mmol/L.")

            # Live platelet flag
            pv = st.session_state.get("plt_input")
            if pv is not None:
                pclr, plbl, pmsg = platelet_interp(pv)
                st.markdown(f"""
                <div class="plt-flag" style="background:{pclr}18;border:1px solid {pclr}55;">
                  <strong style="color:{pclr};">Platelet: {plbl}</strong><br>
                  <span style="color:{T["text_secondary"]};font-size:11px;">{pmsg}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="sec-ttl">📐 CCI Age Contribution</div>', unsafe_allow_html=True)
            av = st.session_state.get("age_input")
            if av is not None:
                ap = cci_age_pts(av)
                st.markdown(f"""
                <div style="background:{T["chip_bg"]};border:1px solid {T["chip_border"]};
                            border-radius:10px;padding:14px 18px;">
                  <span style="font-size:13px;color:{T["text_secondary"]};">Age </span>
                  <strong style="color:{T["teal"]};font-size:15px;">{av} yrs</strong>
                  <span style="color:{T["text_secondary"]};"> → CCI age pts: </span>
                  <strong style="font-family:'JetBrains Mono',monospace;color:{T["teal"]};font-size:22px;">+{ap}</strong>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:{T["chip_bg"]};border:1px solid {T["border"]};
                            border-radius:10px;padding:14px 18px;font-size:13px;color:{T["text_secondary"]};">
                  Enter age above to see CCI age contribution.</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Validate step 1
        s1_errors = []
        if st.session_state.get("age_input")  is None:          s1_errors.append("Age")
        if st.session_state.get("ca_input",  "Not selected") == "Not selected": s1_errors.append("Cardiac arrest")
        if st.session_state.get("mv_input",  "Not selected") == "Not selected": s1_errors.append("Mechanical ventilation")
        if st.session_state.get("cfs_input", "— Select CFS score —") == "— Select CFS score —": s1_errors.append("Clinical Frailty Scale")
        if st.session_state.get("inr_input") is None:            s1_errors.append("INR")
        if st.session_state.get("plt_input") is None:            s1_errors.append("Platelet count")
        if st.session_state.get("hct_input") is None:            s1_errors.append("Hematocrit")
        if st.session_state.get("cl_input")  is None:            s1_errors.append("Chloride")

        if s1_errors:
            st.markdown(f"""
            <div style="background:{T["err_bg"]};border:1px solid {T["err_border"]};
                        border-radius:10px;padding:12px 18px;font-size:13px;margin-bottom:14px;">
              <strong style="color:#f43f5e;">⚠️ Please complete:</strong>
              <span style="color:{T["text_secondary"]};"> {" · ".join(s1_errors)}</span>
            </div>""", unsafe_allow_html=True)

        nav_col, _, hint_col = st.columns([2, 4, 3])
        with nav_col:
            if st.button("Next  →  Comorbidities", type="primary",
                         use_container_width=True, key="s1_next",
                         disabled=bool(s1_errors)):
                st.session_state.calc_step = 2
                st.rerun()
        with hint_col:
            st.markdown(f"""
            <div class="step-nav-hint" style="text-align:right;padding-top:10px;">
              Step 1 of 3 · Comorbidities next
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Comorbidities (CCI)
    # ════════════════════════════════════════════════════════════════════════
    elif step == 2:
        st.markdown(f"""
        <div style="margin-bottom:20px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:{T["text_primary"]};">
            Step 2 — Charlson Comorbidity Index</div>
          <div style="font-size:14px;color:{T["text_secondary"]};margin-top:4px;">
            Tick all conditions present. CCI score is calculated automatically from age and selected conditions.</div>
        </div>""", unsafe_allow_html=True)

        av2  = st.session_state.get("age_input")
        ca2  = cci_age_pts(av2)
        sel2 = [(c, p) for c, p in CCI_CONDITIONS.items() if st.session_state.get(f"cci_{c}", False)]
        cp2  = sum(p for _, p in sel2)
        ct2  = ca2 + cp2

        st.markdown(f"""
        <div style="display:flex;gap:14px;margin-bottom:24px;flex-wrap:wrap;">
          <div class="cci-badge"><div class="cb-lbl">Age Contribution</div>
            <div class="cb-val" style="color:{T["teal"]};">{ca2} pts</div>
            <div class="cb-sub">Auto from age input</div></div>
          <div class="cci-badge"><div class="cb-lbl">Condition Points</div>
            <div class="cb-val" style="color:{T["feat3"]};">{cp2} pts</div>
            <div class="cb-sub">Selected below</div></div>
          <div class="cci-badge"><div class="cb-lbl">Total CCI Score</div>
            <div class="cb-val" style="color:{T["teal"]};">{ct2}</div>
            <div class="cb-sub">Used in model</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sec-ttl">🩺 Tick All Applicable Comorbid Conditions</div>',
                    unsafe_allow_html=True)
        st.caption("Each condition adds weighted points. Tick all that apply to this patient.")

        items = list(CCI_CONDITIONS.items())
        half  = len(items) // 2 + len(items) % 2
        cc1, cc2 = st.columns(2, gap="large")
        for i, (cond, pts) in enumerate(items):
            with (cc1 if i < half else cc2):
                st.checkbox(f"{cond}  (+{pts})", key=f"cci_{cond}")

        # Live total (re-read after widgets render)
        sel3 = [(c, p) for c, p in CCI_CONDITIONS.items() if st.session_state.get(f"cci_{c}", False)]
        cp3  = sum(p for _, p in sel3)
        ct3  = ca2 + cp3

        st.markdown(f"""
        <div style="margin-top:20px;background:{T["chip_bg"]};border:1.5px solid {T["teal"]};
                    border-radius:12px;padding:16px 22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
          <span style="font-size:13px;color:{T["text_secondary"]};">Calculated CCI:</span>
          <span style="font-family:'Space Grotesk',sans-serif;font-size:34px;font-weight:700;color:{T["teal"]};">{ct3}</span>
          <span style="font-size:13px;color:{T["text_secondary"]};">= Age pts ({ca2}) + Condition pts ({cp3})</span>
          <span style="font-size:11px;color:{T["text_secondary"]};margin-left:auto;">
            CCI = 0 uses log(0.5) floor in model</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        nav_l, nav_mid, nav_r = st.columns([2, 2, 3])
        with nav_l:
            if st.button("←  Back", key="s2_back", use_container_width=True):
                st.session_state.calc_step = 1
                st.rerun()
        with nav_mid:
            if st.button("⚡  Calculate Risk", type="primary",
                         use_container_width=True, key="s2_calc"):
                st.session_state.calc_step = 3
                st.rerun()
        with nav_r:
            st.markdown(f"""
            <div class="step-nav-hint" style="text-align:right;padding-top:10px;">
              Step 2 of 3 · Ready to calculate
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Results
    # ════════════════════════════════════════════════════════════════════════
    elif step == 3:
        # Pull all inputs
        age_r = st.session_state.get("age_input")
        ca_r  = st.session_state.get("ca_input",  "Not selected")
        mv_r  = st.session_state.get("mv_input",  "Not selected")
        inr_r = st.session_state.get("inr_input")
        plt_r = st.session_state.get("plt_input")
        hct_r = st.session_state.get("hct_input")
        cl_r  = st.session_state.get("cl_input")
        cfs_r = st.session_state.get("cfs_input", "— Select CFS score —")
        cci_age_r  = cci_age_pts(age_r)
        cci_cond_r = sum(p for c, p in CCI_CONDITIONS.items() if st.session_state.get(f"cci_{c}", False))
        cci_r      = cci_age_r + cci_cond_r

        # Parse
        ca_n  = 1 if ca_r  == "Yes" else 0
        mv_n  = 1 if mv_r  == "Yes" else 0
        cfs_n = int(cfs_r.split("—")[0].strip())

        # ── Show a brief "Calculating…" spinner for clinical realism ──────
        with st.spinner("⚡  Running Cox PH model…"):
            import time as _t; _t.sleep(0.6)
            comp, exp_s, se_comp, bd = compute(age_r, ca_n, inr_r, mv_n, cci_r, cfs_n, hct_r, cl_r)

        m7,  ci7_lo,  ci7_hi  = mort_at(exp_s, 7),  *mort_ci_95(exp_s, se_comp, 7)
        m14, ci14_lo, ci14_hi = mort_at(exp_s, 14), *mort_ci_95(exp_s, se_comp, 14)
        m30, ci30_lo, ci30_hi = mort_at(exp_s, 30), *mort_ci_95(exp_s, se_comp, 30)
        rl  = classify(m30)
        rc  = RISK_CFG[rl]
        emoji = {"LOW":"🟢","MODERATE":"🟡","HIGH":"🔴","VERY HIGH":"🟣"}[rl]

        st.markdown(f"""
        <div style="margin-bottom:20px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:{T["text_primary"]};">
            Step 3 — Risk Assessment Results</div>
          <div style="font-size:14px;color:{T["text_secondary"]};margin-top:4px;">
            Cox Proportional Hazard model output · 95% confidence intervals via delta method</div>
        </div>""", unsafe_allow_html=True)

        # ── Risk banner ──────────────────────────────────────────────────
        st.markdown(f"""
        <div class="risk-banner" style="background:{rc["bg"]};border-color:{rc["border"]};color:{rc["color"]};">
          <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
            <div style="font-size:52px;line-height:1;flex-shrink:0;">{emoji}</div>
            <div style="flex:1;min-width:220px;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                          opacity:0.75;margin-bottom:6px;">30-Day Mortality Risk Classification</div>
              <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px;">
                <span style="font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;">{rl}</span>
                <span style="background:{rc["color"]};color:#fff;border-radius:20px;
                             padding:4px 16px;font-size:12px;font-weight:700;">{rc["band"]}</span>
              </div>
              <div style="font-size:13px;opacity:0.92;line-height:1.65;">
                <strong>Clinical Recommendation:</strong> {rc["rec"]}</div>
            </div>
            <div style="text-align:right;flex-shrink:0;">
              <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;
                          text-transform:uppercase;opacity:0.7;">30-Day Estimate</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:52px;font-weight:700;line-height:1;">
                {m30:.1f}<span style="font-size:20px;">%</span></div>
              <div style="font-size:11px;opacity:0.7;font-family:'JetBrains Mono',monospace;">
                95% CI: {ci30_lo:.1f}–{ci30_hi:.1f}%</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metric cards ─────────────────────────────────────────────────
        mc = st.columns(4, gap="small")
        for col, (lbl, val, ci, clr, sub, ci_lbl) in zip(mc, [
            ("Composite Score",  f"{comp:.4f}",  "",                              T["teal"],                         "Cox PH log hazard", ""),
            ("7-Day Mortality",  f"{m7:.2f}%",   f"{ci7_lo:.1f}–{ci7_hi:.1f}%",  RISK_CFG[classify(m7)]["color"],   classify(m7),        "95% CI"),
            ("14-Day Mortality", f"{m14:.2f}%",  f"{ci14_lo:.1f}–{ci14_hi:.1f}%",RISK_CFG[classify(m14)]["color"],  classify(m14),       "95% CI"),
            ("30-Day Mortality", f"{m30:.2f}%",  f"{ci30_lo:.1f}–{ci30_hi:.1f}%",rc["color"],                       rl,                  "95% CI"),
        ]):
            with col:
                st.markdown(f"""
                <div class="mchip" style="border-color:{clr}44;">
                  <div class="ml">{lbl}</div>
                  <div class="mv" style="color:{clr};">{val}</div>
                  <div class="ms" style="color:{clr};">{sub}</div>
                  {"<div class='ci'>"+ci_lbl+": "+ci+"</div>" if ci else ""}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts ───────────────────────────────────────────────────────
        cha, chb = st.columns([3, 2], gap="large")
        pp, pb, pg, pf = T["plot_paper"], T["plot_bg"], T["plot_grid"], T["plot_font"]

        with cha:
            st.markdown(f"""<div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
                font-weight:700;color:{T["text_primary"]};margin-bottom:10px;">
                📈 Mortality Risk Curve with 95% CI — Days 1–30</div>""",
                unsafe_allow_html=True)
            dx    = list(range(1, 31))
            dy    = [mort_at(exp_s, d) for d in dx]
            dy_lo = [mort_ci_95(exp_s, se_comp, d)[0] for d in dx]
            dy_hi = [mort_ci_95(exp_s, se_comp, d)[1] for d in dx]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dx + dx[::-1], y=dy_hi + dy_lo[::-1],
                fill="toself", fillcolor=rc["bg"].replace("0.09","0.20"),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False, name="95% CI"))
            fig.add_trace(go.Scatter(
                x=dx, y=dy, mode="lines",
                line=dict(color=rc["color"], width=2.5),
                name="Mortality Risk",
                hovertemplate="Day %{x}<br>Risk: %{y:.2f}%<extra></extra>"))
            for kd, kn in [(7,"D7"),(14,"D14"),(30,"D30")]:
                kv = mort_at(exp_s, kd)
                fig.add_vline(x=kd, line_dash="dot",
                              line_color=T["vline"], line_width=1)
                fig.add_annotation(x=kd, y=kv + max(dy)*0.09,
                    text=f"<b>{kn}: {kv:.1f}%</b>",
                    showarrow=False, font=dict(size=11, color=rc["color"]))
            fig.update_layout(
                xaxis=dict(title="ICU Day", color=pf, gridcolor=pg, showgrid=True, linecolor=pg),
                yaxis=dict(title="Mortality Risk (%)", color=pf, gridcolor=pg,
                           range=[0, max(max(dy_hi)*1.25, 5)]),
                height=310, margin=dict(l=40,r=20,t=10,b=40),
                paper_bgcolor=pp, plot_bgcolor=pb,
                legend=dict(font=dict(color=pf, size=11), bgcolor="rgba(0,0,0,0)"),
                font=dict(family="DM Sans", color=pf))
            st.plotly_chart(fig, use_container_width=True)

        with chb:
            st.markdown(f"""<div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
                font-weight:700;color:{T["text_primary"]};margin-bottom:10px;">
                🧩 Factor Score Contributions</div>""", unsafe_allow_html=True)
            sb = pd.DataFrame([
                {"Factor":k, "Score":v, "C":"#f43f5e" if v>=0 else "#10b981"}
                for k,v in bd.items()
            ]).sort_values("Score", ascending=True)
            fig2 = go.Figure(go.Bar(
                x=sb["Score"], y=sb["Factor"], orientation="h",
                marker_color=sb["C"],
                text=sb["Score"].apply(lambda x: f"{x:+.3f}"),
                textposition="outside",
                textfont=dict(size=10, family="JetBrains Mono", color=pf),
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
            fig2.update_layout(
                xaxis=dict(title="Score Contribution", color=pf, gridcolor=pg, linecolor=pg),
                yaxis=dict(color=pf),
                height=310, margin=dict(l=10,r=60,t=10,b=40),
                paper_bgcolor=pp, plot_bgcolor=pb,
                showlegend=False, font=dict(family="DM Sans", color=pf))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Score breakdown table ─────────────────────────────────────────
        st.markdown(f"""<div style="font-family:'Space Grotesk',sans-serif;font-size:13px;
            font-weight:700;color:{T["text_primary"]};margin:4px 0 12px 0;">
            🔬 Cox PH Score Breakdown</div>""", unsafe_allow_html=True)
        bl    = list(BETA.values())
        bl_se = list(BETA_SE.values())
        rows = [{"Factor":k, "β":f"{bl[i]:+.5f}", "SE(β)":f"±{bl_se[i]:.4f}", "Contribution":f"{v:+.6f}"}
                for i,(k,v) in enumerate(bd.items())]
        rows.append({"Factor":"▶  COMPOSITE SCORE (Σ)", "β":"—",
                     "SE(β)":f"±{se_comp:.4f}", "Contribution":f"{comp:+.6f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # ── Platelet flag ─────────────────────────────────────────────────
        if plt_r is not None:
            pclr, plbl, pmsg = platelet_interp(plt_r)
            st.markdown(f"""
            <div class="plt-flag" style="background:{pclr}18;border:1px solid {pclr}55;margin-top:8px;">
              <strong style="color:{pclr};">Platelet Count — {plbl}</strong><br>
              <span style="color:{T["text_secondary"]};font-size:12px;">{pmsg}
              <em>(contextual flag — not included in Cox model score)</em></span>
            </div>""", unsafe_allow_html=True)

        # ── Full day table ────────────────────────────────────────────────
        with st.expander("📋 Full Day-by-Day Survival & Mortality Table"):
            trows = []
            for d in sorted(BASELINE_HAZARD):
                m = mort_at(exp_s, d)
                lo, hi = mort_ci_95(exp_s, se_comp, d)
                trows.append({"Day":d, "Survival (%)":f"{100-m:.3f}",
                               "Mortality (%)":f"{m:.3f}",
                               "95% CI Lo":f"{lo:.3f}", "95% CI Hi":f"{hi:.3f}",
                               "Band":classify(m)})
            st.dataframe(pd.DataFrame(trows), hide_index=True, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Bottom navigation ─────────────────────────────────────────────
        bot_l, bot_m, bot_r = st.columns([2, 2, 3])
        with bot_l:
            if st.button("←  Back to CCI", key="s3_back", use_container_width=True):
                st.session_state.calc_step = 2
                st.rerun()
        with bot_m:
            if st.button("🔄  New Patient", type="secondary",
                         use_container_width=True, key="new_pt"):
                keep = {"page", "dark_mode"}
                for k in list(st.session_state.keys()):
                    if k not in keep:
                        del st.session_state[k]
                st.rerun()

        st.markdown(f"""
        <div style="margin-top:22px;background:{T["warn_bg"]};border:1px solid {T["warn_border"]};
                    border-radius:10px;padding:14px 18px;font-size:12px;
                    color:{T["text_secondary"]};line-height:1.7;">
          ⚠️ <strong style="color:{T["warn_text"]};">Clinical Disclaimer:</strong>
          Decision-support aid for trained clinical professionals only. Does not replace multidisciplinary
          team judgement. All outputs are probabilistic estimates with inherent uncertainty (shown as 95% CI).
          <strong>Version 2.0 — September 2025.</strong>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "about":
    bk2, _ = st.columns([1, 9])
    with bk2:
        if st.button("← Home", key="bk_a"):
            st.session_state.page = "home"; st.rerun()

    st.markdown(f"""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;
                color:{T["text_primary"]};margin-bottom:4px;">About This Tool</div>
    <div style="font-size:14px;color:{T["text_secondary"]};margin-bottom:26px;">
        Methodology, model variables, accuracy improvements, and clinical interpretation guide.</div>""",
        unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(f"""
        <div class="icu-card">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      color:{T["teal"]};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:14px;">
            ⚙️ Model &amp; Accuracy Improvements (v2.0)</div>
          <p style="font-size:14px;color:{T["text_secondary"]};line-height:1.8;">
            Implements a <strong style="color:{T["text_primary"]};">Cox Proportional Hazard (CPH) model</strong>
            validated for elderly ICU patients (≥ 65 years). Version 2.0 introduces:</p>
          <ul style="font-size:13px;color:{T["text_secondary"]};line-height:2;padding-left:18px;">
            <li><strong style="color:{T["text_primary"]};">95% Confidence Intervals</strong> via delta method on composite score</li>
            <li><strong style="color:{T["text_primary"]};">Log floor fix:</strong> CCI=0 uses log(0.5) instead of log(0), preventing model singularity for healthy patients</li>
            <li><strong style="color:{T["text_primary"]};">INR floor:</strong> clamped to 0.5 minimum, preventing distortion from near-zero values</li>
            <li><strong style="color:{T["text_primary"]};">Platelet contextual flags:</strong> clinical severity alerts (not in CPH score) for complete bedside picture</li>
            <li><strong style="color:{T["text_primary"]};">SE propagation:</strong> coefficient standard errors carried through to CI calculation</li>
          </ul>
          <div style="background:{T["chip_bg"]};border:1px solid {T["chip_border"]};border-radius:8px;
                      padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:13px;color:{T["teal"]};margin:14px 0;">
            S(t) = exp(−H₀(t) × exp(Σ βᵢXᵢ))</div>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown(f"""
        <div class="icu-card">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      color:{T["teal"]};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:14px;">
            📐 Variables, β Coefficients &amp; SE</div>""", unsafe_allow_html=True)
        for var, typ, beta, se, note in [
            ("Cardiac Arrest",         "Binary (0/1)", "+2.7268", "±0.412", "Strongest predictor"),
            ("Loge INR",               "ln(INR≥0.5)", "+1.7000", "±0.298", "Coagulopathy"),
            ("Mechanical Ventilation", "Binary (0/1)", "+0.9076", "±0.187", "Respiratory failure"),
            ("Loge Age",               "ln(Age)",      "+1.5523", "±0.341", "Age-adjusted risk"),
            ("Loge CCI",               "ln(CCI≥0.5)", "+0.8597", "±0.203", "Comorbidity burden"),
            ("CFS Score",              "1–8 scale",    "+0.0598", "±0.028", "Frailty"),
            ("Hematocrit",             "%",            "−0.0583", "±0.018", "Anaemia / perfusion"),
            ("Chloride",               "mmol/L",       "−0.00096","±0.00042","Electrolyte balance"),
        ]:
            clr = "#f43f5e" if "+" in beta else "#10b981"
            st.markdown(f"""
            <div class="coeff-row">
              <div style="flex:2;font-size:12px;color:{T["text_primary"]};font-weight:500;">{var}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{clr};font-weight:700;min-width:68px;">{beta}</div>
              <div style="font-size:11px;color:{T["text_secondary"]};min-width:62px;">{se}</div>
              <div style="flex:2;font-size:11px;color:{T["text_secondary"]};">{note}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="icu-card">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                  color:{T["teal"]};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:18px;">
        🏥 Risk Band Clinical Guidance</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">""",
        unsafe_allow_html=True)
    for rl, cfg in RISK_CFG.items():
        st.markdown(f"""
        <div style="background:{cfg["bg"]};border:1px solid {cfg["border"]};border-radius:12px;padding:16px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;color:{cfg["color"]};margin-bottom:4px;">{rl}</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:{cfg["color"]};margin-bottom:8px;">{cfg["band"]}</div>
          <div style="font-size:12px;color:{T["text_secondary"]};line-height:1.6;">{cfg["rec"]}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬  Start Patient Assessment", type="primary"):
        st.session_state.page = "calculator"; st.rerun()
