"""
Elderly ICU Mortality Risk Calculator – Version 1.0 (Sept 2025)
Award-Grade Streamlit Application | Cox Proportional Hazard Model
Perfect Dark / Light Mode Toggle
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

for key, default in [("dark_mode", True), ("page", "home")]:
    if key not in st.session_state:
        st.session_state[key] = default

DARK = dict(
    page_grad="radial-gradient(ellipse 140% 80% at 8% -8%, #0d2a4a 0%, #05101f 65%)",
    page_bg="#05101f", card_bg="rgba(10,31,56,0.82)", card_bg_solid="#0a1f38",
    input_bg="rgba(16,45,80,0.92)", input_sel_bg="#0a1f38", radio_bg="rgba(16,45,80,0.65)",
    popover_bg="#0a1f38", stat_bg="rgba(10,31,56,0.82)", chip_bg="rgba(0,201,177,0.08)",
    warn_bg="rgba(245,158,11,0.07)", err_bg="rgba(244,63,94,0.08)",
    expander_cont="rgba(5,16,31,0.55)",
    text_primary="#f0f6ff", text_secondary="#8fa3b8", text_input="#ffffff",
    text_label="#00c9b1", text_placeholder="rgba(143,163,184,0.45)", warn_text="#fcd34d",
    border="rgba(0,201,177,0.18)", border_focus="#00c9b1",
    chip_border="rgba(0,201,177,0.28)", warn_border="rgba(245,158,11,0.28)",
    err_border="rgba(244,63,94,0.38)",
    teal="#00c9b1", teal_glow="rgba(0,201,177,0.14)", teal_dim="#00a898",
    btn_grad_end="#0096d6", btn_text="#05101f", shadow_btn="rgba(0,201,177,0.38)",
    plot_paper="rgba(10,31,56,0.82)", plot_bg="rgba(5,16,31,0.45)",
    plot_grid="rgba(143,163,184,0.10)", plot_font="#8fa3b8",
    toggle_icon="☀️", toggle_label="Light Mode",
    tab_inactive="#8fa3b8", tab_active_bg="rgba(0,201,177,0.12)",
    feat2="60b8e8", feat3="a78bfa", feat4="fcd34d",
)
LIGHT = dict(
    page_grad="linear-gradient(160deg, #daeeff 0%, #eef3fb 55%)",
    page_bg="#eef3fb", card_bg="rgba(255,255,255,0.96)", card_bg_solid="#ffffff",
    input_bg="#ffffff", input_sel_bg="#ffffff", radio_bg="rgba(230,240,255,0.80)",
    popover_bg="#ffffff", stat_bg="rgba(255,255,255,0.96)", chip_bg="rgba(0,140,130,0.07)",
    warn_bg="rgba(215,130,0,0.07)", err_bg="rgba(210,30,30,0.06)",
    expander_cont="rgba(238,243,251,0.85)",
    text_primary="#09213a", text_secondary="#4d6a87", text_input="#09213a",
    text_label="#006d62", text_placeholder="rgba(77,106,135,0.45)", warn_text="#a05c00",
    border="rgba(0,140,130,0.20)", border_focus="#00897b",
    chip_border="rgba(0,140,130,0.25)", warn_border="rgba(215,130,0,0.30)",
    err_border="rgba(210,30,30,0.30)",
    teal="#00897b", teal_glow="rgba(0,137,123,0.10)", teal_dim="#00695c",
    btn_grad_end="#0277bd", btn_text="#ffffff", shadow_btn="rgba(0,137,123,0.28)",
    plot_paper="rgba(255,255,255,0.96)", plot_bg="rgba(243,248,255,0.80)",
    plot_grid="rgba(77,106,135,0.10)", plot_font="#4d6a87",
    toggle_icon="🌙", toggle_label="Dark Mode",
    tab_inactive="#4d6a87", tab_active_bg="rgba(0,137,123,0.10)",
    feat2="0277bd", feat3="6d28d9", feat4="a05c00",
)

T = DARK if st.session_state.dark_mode else LIGHT

def css(t):
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --teal:{t['teal']};--teal-glow:{t['teal_glow']};
  --tp:{t['text_primary']};--ts:{t['text_secondary']};
  --border:{t['border']};--card:{t['card_bg']};
  --Fh:'Space Grotesk',sans-serif;--Fb:'DM Sans',sans-serif;--Fm:'JetBrains Mono',monospace;
}}
html,body{{background:{t['page_bg']} !important;}}
.stApp{{background:{t['page_grad']} !important;min-height:100vh;font-family:var(--Fb) !important;color:{t['text_primary']} !important;}}
#MainMenu,footer,header{{visibility:hidden !important;}}
[data-testid="stSidebar"]{{display:none !important;}}
.block-container{{padding:0 2rem 4rem 2rem !important;max-width:1320px !important;}}
h1,h2,h3,h4,h5{{font-family:var(--Fh) !important;color:{t['text_primary']} !important;}}
p,li{{color:{t['text_primary']} !important;}}
div[data-testid="stWidgetLabel"] p,div[data-testid="stWidgetLabel"] label,
.stNumberInput label,.stSelectbox label,.stRadio > label,.stSlider label,.stTextInput label{{
  color:{t['text_label']} !important;font-family:var(--Fb) !important;font-size:12px !important;
  font-weight:700 !important;letter-spacing:0.05em !important;text-transform:uppercase !important;}}
.stNumberInput input,.stTextInput input{{
  background:{t['input_bg']} !important;border:1.5px solid {t['border']} !important;
  border-radius:10px !important;color:{t['text_input']} !important;
  font-family:var(--Fm) !important;font-size:15px !important;font-weight:600 !important;
  padding:10px 14px !important;caret-color:{t['teal']} !important;}}
.stNumberInput input:focus,.stTextInput input:focus{{
  border-color:{t['border_focus']} !important;box-shadow:0 0 0 3px {t['teal_glow']} !important;outline:none !important;}}
.stNumberInput input::placeholder,.stTextInput input::placeholder{{color:{t['text_placeholder']} !important;}}
.stSelectbox > div > div{{background:{t['input_bg']} !important;border:1.5px solid {t['border']} !important;border-radius:10px !important;color:{t['text_input']} !important;}}
.stSelectbox > div > div:focus-within{{border-color:{t['border_focus']} !important;}}
[data-baseweb="select"] > div{{background:{t['input_bg']} !important;color:{t['text_input']} !important;}}
[data-baseweb="select"] span{{color:{t['text_input']} !important;}}
[data-baseweb="popover"],[data-baseweb="popover"] *{{background:{t['popover_bg']} !important;color:{t['text_input']} !important;}}
[data-baseweb="menu"]{{background:{t['popover_bg']} !important;}}
[data-baseweb="option"]{{color:{t['text_input']} !important;background:{t['popover_bg']} !important;}}
[data-baseweb="option"]:hover{{background:{t['teal_glow']} !important;}}
.stRadio > div{{gap:8px !important;}}
div[data-testid="stRadio"] label{{
  background:{t['radio_bg']} !important;border:1.5px solid {t['border']} !important;
  border-radius:8px !important;padding:8px 18px !important;cursor:pointer !important;
  color:{t['text_primary']} !important;font-size:13px !important;font-weight:500 !important;
  text-transform:none !important;letter-spacing:0 !important;}}
div[data-testid="stRadio"] label:has(input:checked){{
  background:{t['teal_glow']} !important;border-color:{t['teal']} !important;color:{t['teal']} !important;}}
.stCheckbox label{{color:{t['text_primary']} !important;font-size:13px !important;font-weight:400 !important;text-transform:none !important;letter-spacing:0 !important;}}
.stButton > button{{font-family:var(--Fh) !important;font-weight:700 !important;font-size:14px !important;letter-spacing:0.05em !important;border-radius:12px !important;cursor:pointer !important;}}
.stButton > button[kind="primary"]{{background:linear-gradient(135deg,{t['teal']} 0%,{t['btn_grad_end']} 100%) !important;color:{t['btn_text']} !important;border:none !important;padding:13px 28px !important;box-shadow:0 4px 18px {t['shadow_btn']} !important;}}
.stButton > button[kind="primary"]:hover{{transform:translateY(-2px) !important;box-shadow:0 8px 26px {t['shadow_btn']} !important;}}
.stButton > button[kind="secondary"]{{background:transparent !important;border:1.5px solid {t['border']} !important;color:{t['text_secondary']} !important;padding:9px 22px !important;}}
.stButton > button[kind="secondary"]:hover{{border-color:{t['teal']} !important;color:{t['teal']} !important;}}
.stTabs [data-baseweb="tab-list"]{{background:transparent !important;border-bottom:1px solid {t['border']} !important;gap:2px !important;}}
.stTabs [data-baseweb="tab"]{{font-family:var(--Fh) !important;font-size:12px !important;font-weight:700 !important;letter-spacing:0.07em !important;color:{t['tab_inactive']} !important;background:transparent !important;border:none !important;padding:12px 20px !important;border-radius:8px 8px 0 0 !important;text-transform:uppercase !important;}}
.stTabs [aria-selected="true"]{{color:{t['teal']} !important;background:{t['tab_active_bg']} !important;border-bottom:2px solid {t['teal']} !important;}}
.stTabs [data-baseweb="tab-panel"]{{background:transparent !important;}}
.streamlit-expanderHeader{{background:{t['card_bg']} !important;border:1px solid {t['border']} !important;border-radius:10px !important;color:{t['text_primary']} !important;font-family:var(--Fh) !important;font-size:12px !important;font-weight:700 !important;}}
.streamlit-expanderContent{{background:{t['expander_cont']} !important;border:1px solid {t['border']} !important;border-top:none !important;border-radius:0 0 10px 10px !important;}}
.stDataFrame{{border:1px solid {t['border']} !important;border-radius:10px !important;overflow:hidden !important;}}
[data-testid="stDataFrameResizable"]{{background:{t['card_bg_solid']} !important;}}
.stDataFrame th{{background:{t['card_bg']} !important;color:{t['text_label']} !important;font-size:11px !important;letter-spacing:0.07em !important;text-transform:uppercase !important;}}
.stDataFrame td{{color:{t['text_primary']} !important;font-size:13px !important;}}
hr{{border-color:{t['border']} !important;margin:8px 0 22px 0 !important;}}
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:{t['border']};border-radius:4px;}}
.icu-logo-ring{{width:42px;height:42px;background:linear-gradient(135deg,{t['teal']},{t['btn_grad_end']});border-radius:12px;display:grid;place-items:center;font-size:22px;flex-shrink:0;box-shadow:0 4px 16px {t['shadow_btn']};}}
.icu-nav-title{{font-family:var(--Fh);font-size:17px;font-weight:700;color:{t['text_primary']};}}
.icu-nav-sub{{font-size:11px;color:{t['text_secondary']};letter-spacing:0.08em;text-transform:uppercase;margin-top:2px;}}
.icu-badge{{background:{t['chip_bg']};border:1px solid {t['teal']};border-radius:20px;padding:5px 14px;font-size:11px;font-family:var(--Fm);color:{t['teal']};letter-spacing:0.04em;white-space:nowrap;}}
.icu-card{{background:{t['card_bg']};border:1px solid {t['border']};border-radius:16px;padding:24px;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);}}
.icu-stat{{background:{t['stat_bg']};border:1px solid {t['border']};border-radius:14px;padding:20px 22px;position:relative;overflow:hidden;}}
.icu-stat::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,{t['teal']},transparent);}}
.icu-stat .lbl{{font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{t['text_secondary']};margin-bottom:8px;}}
.icu-stat .val{{font-family:var(--Fh);font-size:30px;font-weight:700;color:{t['text_primary']};line-height:1;}}
.icu-stat .sub{{font-size:12px;color:{t['text_secondary']};margin-top:6px;}}
.sec-title{{font-family:var(--Fh);font-size:12px;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;color:{t['teal']};padding:10px 0 8px 0;border-bottom:1px solid {t['border']};margin-bottom:16px;display:flex;align-items:center;gap:8px;}}
.risk-banner{{border-radius:16px;padding:24px 28px;border:1px solid;position:relative;overflow:hidden;}}
.risk-banner::after{{content:'';position:absolute;bottom:-40px;right:-30px;width:160px;height:160px;border-radius:50%;background:radial-gradient(circle,currentColor 0%,transparent 70%);opacity:0.06;pointer-events:none;}}
.metric-chip{{border-radius:14px;padding:18px 20px;border:1px solid;position:relative;overflow:hidden;background:{t['stat_bg']};}}
.mc-label{{font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;color:{t['text_secondary']};}}
.mc-val{{font-family:var(--Fh);font-size:26px;font-weight:700;line-height:1;}}
.mc-sub{{font-size:11px;font-weight:600;margin-top:5px;opacity:0.85;}}
.feat-chip{{background:{t['chip_bg']};border:1px solid {t['chip_border']};border-radius:20px;padding:6px 14px;font-size:12px;font-weight:600;display:inline-block;font-family:var(--Fb);}}
.ref-row{{display:flex;align-items:flex-start;gap:12px;padding:11px 0;border-bottom:1px solid {t['border']};}}
.coeff-row{{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid {t['border']};}}
.cci-badge{{background:{t['chip_bg']};border:1px solid {t['chip_border']};border-radius:10px;padding:14px 20px;flex:1;min-width:150px;}}
.cb-lbl{{font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{t['text_secondary']};margin-bottom:6px;}}
.cb-val{{font-family:var(--Fh);font-size:28px;font-weight:700;line-height:1;}}
.cb-sub{{font-size:12px;color:{t['text_secondary']};margin-top:4px;}}
</style>"""

st.markdown(css(T), unsafe_allow_html=True)

# ── Constants ──
BETA = {"cardiac_arrest":2.7268,"log_inr":1.7000,"mechanical_vent":0.9076,
        "log_age":1.55229,"log_cci":0.8597,"cfs":0.05976,"hematocrit":-0.05827,"chloride":-0.00096}
BASELINE_HAZARD = {1:7e-6,2:1.5e-5,3:1.9e-5,4:3.8e-5,5:8.2e-5,6:9.9e-5,7:1.09e-4,8:1.32e-4,
    9:1.45e-4,10:1.59e-4,11:1.79e-4,12:2.0e-4,13:2.0e-4,14:2.0e-4,15:2.48e-4,16:2.75e-4,
    17:3.34e-4,18:3.67e-4,19:4.03e-4,20:4.42e-4,21:4.42e-4,22:4.91e-4,24:5.61e-4,26:6.52e-4,30:7.56e-4}
CCI_CONDITIONS = {"Myocardial infarction":1,"Congestive heart failure":1,"Peripheral vascular disease":1,
    "Cerebrovascular accident / TIA":1,"Dementia":1,"Chronic pulmonary disease":1,
    "Connective tissue disease":1,"Peptic ulcer disease":1,"Mild liver disease":1,
    "Uncomplicated diabetes":1,"Hemiplegia":2,"Moderate to severe CKD":2,
    "Diabetes with end-organ damage":2,"Localized solid tumor":2,"Leukemia":2,"Lymphoma":2,
    "Moderate to severe liver disease":3,"Metastatic solid tumor":6,"AIDS":6}
CFS_OPTIONS = {1:"1 — Very Fit",2:"2 — Well",3:"3 — Managing Well",4:"4 — Vulnerable",
    5:"5 — Mildly Frail",6:"6 — Moderately Frail",7:"7 — Severely Frail",8:"8 — Very Severely Frail"}
RISK_CFG = {
    "LOW":      dict(band="< 10%", color="#10b981",bg="rgba(16,185,129,0.09)",border="rgba(16,185,129,0.40)",
                     rec="ICU admission strongly recommended. Anticipate favourable outcome with standard monitoring protocols."),
    "MODERATE": dict(band="10–30%",color="#f59e0b",bg="rgba(245,158,11,0.09)", border="rgba(245,158,11,0.40)",
                     rec="ICU admission appropriate. Ensure early rehabilitation, daily goal-setting, and proactive family communication."),
    "HIGH":     dict(band="30–60%",color="#f43f5e",bg="rgba(244,63,94,0.09)",  border="rgba(244,63,94,0.40)",
                     rec="Discuss in triage rounds. Initiate family counselling, advance care planning, and palliative care consultation."),
    "VERY HIGH":dict(band="> 60%", color="#c026d3",bg="rgba(192,38,211,0.09)", border="rgba(192,38,211,0.40)",
                     rec="Case-by-case ICU consideration. Strong recommendation for goals-of-care discussion and palliative care."),
}

def classify(p):
    return "LOW" if p<10 else "MODERATE" if p<=30 else "HIGH" if p<=60 else "VERY HIGH"

def mort_at(es,day):
    if day in BASELINE_HAZARD: return (1-np.exp(-BASELINE_HAZARD[day]*es))*100
    sd=sorted(BASELINE_HAZARD)
    for i in range(len(sd)-1):
        d1,d2=sd[i],sd[i+1]
        if d1<=day<=d2:
            h=BASELINE_HAZARD[d1]+(day-d1)/(d2-d1)*(BASELINE_HAZARD[d2]-BASELINE_HAZARD[d1])
            return (1-np.exp(-h*es))*100
    return (1-np.exp(-BASELINE_HAZARD[30]*es))*100

def compute(age,ca,inr,mv,cci,cfs,hct,cl):
    la=np.log(max(age,1));li=np.log(max(inr,1e-4));lc=np.log(max(cci,1e-4))
    c=(BETA["cardiac_arrest"]*ca+BETA["log_inr"]*li+BETA["mechanical_vent"]*mv
       +BETA["log_age"]*la+BETA["log_cci"]*lc+BETA["cfs"]*cfs+BETA["hematocrit"]*hct+BETA["chloride"]*cl)
    es=np.exp(c)
    bd={"Cardiac Arrest":BETA["cardiac_arrest"]*ca,"Loge INR":BETA["log_inr"]*li,
        "Mechanical Ventilation":BETA["mechanical_vent"]*mv,"Loge Age":BETA["log_age"]*la,
        "Loge CCI":BETA["log_cci"]*lc,"Clinical Frailty Scale":BETA["cfs"]*cfs,
        "Hematocrit":BETA["hematocrit"]*hct,"Chloride":BETA["chloride"]*cl}
    return c,es,bd

def cci_age(a): return 0 if a is None else (1 if a<=69 else (2 if a<=79 else 4))

# ── Nav ──
nl,nm,nr = st.columns([5,2,2])
with nl:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:12px;padding:14px 0 10px 0;">
      <div class="icu-logo-ring">🫀</div>
      <div><div class="icu-nav-title">ICU Mortality Risk Calculator</div>
           <div class="icu-nav-sub">Elderly Patient · Cox Proportional Hazard Model</div></div>
    </div>""", unsafe_allow_html=True)
with nm:
    st.markdown(f'<div style="padding-top:18px;"><span class="icu-badge">v1.0 · Sept 2025</span></div>',unsafe_allow_html=True)
with nr:
    # Pill toggle — uses a hidden Streamlit button triggered by JS click on the pill
    is_dark = st.session_state.dark_mode
    pill_bg        = "#0a1f38"   if is_dark else "#e2eaf4"
    pill_border    = "#00c9b1"   if is_dark else "#00897b"
    thumb_bg       = "#00c9b1"   if is_dark else "#00897b"
    thumb_pos      = "calc(100% - 28px)" if is_dark else "2px"
    label_color    = "#8fa3b8"   if is_dark else "#4d6a87"
    icon_left      = "🌙"
    icon_right     = "☀️"
    active_icon_color = "#05101f" if is_dark else "#ffffff"
    st.markdown(f"""
    <style>
    #theme-pill-wrap {{ display:flex; align-items:center; justify-content:flex-end; padding-top:14px; gap:10px; }}
    #theme-pill {{
        width:56px; height:28px; border-radius:14px;
        background:{pill_bg}; border:1.5px solid {pill_border};
        position:relative; cursor:pointer;
        transition:background 0.3s ease, border-color 0.3s ease;
        flex-shrink:0;
    }}
    #theme-thumb {{
        width:24px; height:24px; border-radius:50%;
        background:{thumb_bg};
        position:absolute; top:0px; left:{thumb_pos};
        transition:left 0.25s ease, background 0.25s ease;
        display:grid; place-items:center;
        font-size:13px; line-height:1;
        box-shadow:0 2px 6px rgba(0,0,0,0.25);
    }}
    #theme-label {{
        font-family:'DM Sans',sans-serif; font-size:12px; font-weight:600;
        color:{label_color}; white-space:nowrap; letter-spacing:0.03em;
    }}
    </style>
    <div id="theme-pill-wrap">
        <span id="theme-label">{"Dark" if is_dark else "Light"} Mode</span>
        <div id="theme-pill" onclick="document.getElementById('theme-toggle-btn').click()">
            <div id="theme-thumb">{"🌙" if is_dark else "☀️"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Hidden real Streamlit button wired to the pill via JS
    if st.button("toggle", key="theme_btn", label_visibility="collapsed"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    # Wire JS id to actual Streamlit button
    st.markdown("""
    <script>
    (function() {
        const btn = Array.from(window.parent.document.querySelectorAll('button'))
            .find(b => b.innerText.trim() === 'toggle');
        if (btn) btn.id = 'theme-toggle-btn';
    })();
    </script>
    """, unsafe_allow_html=True)

st.markdown("<hr>",unsafe_allow_html=True)

page = st.session_state.page

# ════════════════════════════════════════════ HOME ═══
if page=="home":
    hl,hr=st.columns([3,2],gap="large")
    with hl:
        st.markdown(f"""<div style="padding:6px 0 18px 0;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      letter-spacing:0.14em;text-transform:uppercase;color:{T['teal']};margin-bottom:14px;">
            CLINICAL DECISION SUPPORT TOOL</div>
          <h1 style="font-family:'Space Grotesk',sans-serif;font-size:40px;font-weight:700;line-height:1.18;margin:0 0 16px 0;">
            Predict ICU Mortality<br><span style="color:{T['teal']};">With Precision.</span></h1>
          <p style="font-size:15px;color:{T['text_secondary']};line-height:1.75;max-width:520px;margin-bottom:26px;">
            A validated prognostic tool for clinicians managing elderly ICU admissions (≥ 65 years).
            Computes 7, 14 and 30-day mortality risk using the Cox Proportional Hazard model,
            with instant risk stratification and evidence-based clinical guidance.</p>
        </div>""",unsafe_allow_html=True)
        b1,b2=st.columns([2,1])
        with b1:
            if st.button("🔬  Begin Patient Assessment",type="primary",use_container_width=True):
                st.session_state.page="calculator";st.rerun()
        with b2:
            if st.button("📖  How It Works",use_container_width=True):
                st.session_state.page="about";st.rerun()
        st.markdown(f"""<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:22px;">
          <span class="feat-chip" style="color:{T['teal']};">⚡ Instant Results</span>
          <span class="feat-chip" style="color:#{T['feat2']};">📊 Survival Curves</span>
          <span class="feat-chip" style="color:#{T['feat3']};">🏥 Clinical Guidance</span>
          <span class="feat-chip" style="color:#{T['feat4']};">🔢 CCI Calculator</span>
        </div>""",unsafe_allow_html=True)
    with hr:
        st.markdown(f"""<div class="icu-card" style="margin-top:8px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      color:{T['teal']};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:16px;">
            📋 Risk Band Reference</div>""",unsafe_allow_html=True)
        for rl,cfg in RISK_CFG.items():
            st.markdown(f"""<div class="ref-row">
              <div style="width:10px;height:10px;border-radius:50%;background:{cfg['color']};
                          box-shadow:0 0 8px {cfg['color']};flex-shrink:0;margin-top:5px;"></div>
              <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;color:{cfg['color']};">
                  {rl} <span style="color:{T['text_secondary']};font-weight:400;font-size:12px;">({cfg['band']})</span></div>
                <div style="font-size:12px;color:{T['text_secondary']};margin-top:3px;line-height:1.5;">{cfg['rec']}</div>
              </div></div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    sc=st.columns(4,gap="small")
    for col,(lbl,val,sub) in zip(sc,[("Model Variables","8","Clinical parameters"),("Time Points","25","Days 1–30"),
                                      ("Age Threshold","≥ 65","Years old"),("Risk Categories","4","LOW → VERY HIGH")]):
        with col:
            st.markdown(f"""<div class="icu-stat"><div class="lbl">{lbl}</div>
              <div class="val" style="color:{T['teal']};">{val}</div><div class="sub">{sub}</div></div>""",
              unsafe_allow_html=True)

# ════════════════════════════════════════ CALCULATOR ═══
elif page=="calculator":
    bk,_=st.columns([1,9])
    with bk:
        if st.button("← Home",key="bk_c"): st.session_state.page="home";st.rerun()
    st.markdown(f"""<div style="margin-bottom:18px;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:{T['text_primary']};">
        Patient Assessment Form</div>
      <div style="font-size:14px;color:{T['text_secondary']};margin-top:4px;">
        Complete all sections, then open the <strong style="color:{T['teal']};">RESULTS</strong> tab.</div>
    </div>""",unsafe_allow_html=True)

    tab1,tab2,tab3=st.tabs(["  🧑  PATIENT DETAILS  ","  🩺  COMORBIDITIES (CCI)  ","  📊  RESULTS  "])

    with tab1:
        st.markdown("<br>",unsafe_allow_html=True)
        tl,tr=st.columns(2,gap="large")
        with tl:
            st.markdown(f'<div class="sec-title">🧑 Demographics</div>',unsafe_allow_html=True)
            age_v=st.number_input("Age upon ICU admission (years)",min_value=65,max_value=120,step=1,
                value=None,placeholder="Enter age  (≥ 65)",key="age_input",
                help="Must be ≥ 65. Age contributes automatically to CCI.")
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown(f'<div class="sec-title">⚡ Acute Clinical Events</div>',unsafe_allow_html=True)
            ca_v=st.radio("Cardiac arrest prior to ICU admission?",
                ["Not selected","Yes","No"],index=0,horizontal=True,key="ca_input")
            mv_v=st.radio("Mechanically ventilated upon ICU admission?",
                ["Not selected","Yes","No"],index=0,horizontal=True,key="mv_input")
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown(f'<div class="sec-title">🚶 Frailty Assessment</div>',unsafe_allow_html=True)
            cfs_v=st.selectbox("Clinical Frailty Scale (CFS)",
                ["— Select CFS score —"]+[CFS_OPTIONS[i] for i in range(1,9)],index=0,key="cfs_input")
        with tr:
            st.markdown(f'<div class="sec-title">🧪 Laboratory Values</div>',unsafe_allow_html=True)
            inr_v=st.number_input("International Normalised Ratio (INR)",min_value=0.5,max_value=20.0,
                step=0.1,format="%.2f",value=None,placeholder="e.g.  1.20",key="inr_input")
            plt_v=st.number_input("Platelet count  (×10⁹/L)",min_value=1,max_value=1000,step=1,
                value=None,placeholder="e.g.  150",key="plt_input")
            hct_v=st.number_input("Hematocrit  (%)",min_value=1.0,max_value=70.0,step=0.1,format="%.1f",
                value=None,placeholder="e.g.  35.0",key="hct_input")
            cl_v=st.number_input("Chloride  (mmol/L)",min_value=60,max_value=140,step=1,
                value=None,placeholder="e.g.  102",key="cl_input")
            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown(f'<div class="sec-title">📐 CCI Age Contribution</div>',unsafe_allow_html=True)
            av=st.session_state.get("age_input")
            if av is not None:
                ap=cci_age(av)
                st.markdown(f"""<div style="background:{T['chip_bg']};border:1px solid {T['chip_border']};
                    border-radius:10px;padding:14px 18px;">
                  <span style="font-size:13px;color:{T['text_secondary']};">Age </span>
                  <strong style="color:{T['teal']};font-size:15px;">{av} yrs</strong>
                  <span style="color:{T['text_secondary']};font-size:13px;"> → CCI age points: </span>
                  <strong style="font-family:'JetBrains Mono',monospace;color:{T['teal']};font-size:22px;">+{ap}</strong>
                </div>""",unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="background:{T['chip_bg']};border:1px solid {T['border']};
                    border-radius:10px;padding:14px 18px;font-size:13px;color:{T['text_secondary']};">
                  Enter age above to see CCI age contribution.</div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"""<div style="background:{T['warn_bg']};border:1px solid {T['warn_border']};
            border-radius:10px;padding:12px 18px;font-size:13px;color:{T['warn_text']};">
          ℹ️ After completing this tab, proceed to <strong>COMORBIDITIES (CCI)</strong>, then open <strong>RESULTS</strong>.
        </div>""",unsafe_allow_html=True)

    with tab2:
        st.markdown("<br>",unsafe_allow_html=True)
        av2=st.session_state.get("age_input"); ca2=cci_age(av2)
        sel2=[(c,p) for c,p in CCI_CONDITIONS.items() if st.session_state.get(f"cci_{c}",False)]
        cp2=sum(p for _,p in sel2); ct2=ca2+cp2
        st.markdown(f"""<div style="display:flex;gap:14px;margin-bottom:22px;flex-wrap:wrap;">
          <div class="cci-badge"><div class="cb-lbl">Age Contribution</div>
            <div class="cb-val" style="color:{T['teal']};">{ca2} pts</div><div class="cb-sub">Auto from age</div></div>
          <div class="cci-badge"><div class="cb-lbl">Condition Points</div>
            <div class="cb-val" style="color:#{T['feat3']};">{cp2} pts</div><div class="cb-sub">Select below</div></div>
          <div class="cci-badge"><div class="cb-lbl">Total CCI Score</div>
            <div class="cb-val" style="color:{T['teal']};">{ct2}</div><div class="cb-sub">Age + Conditions</div></div>
        </div>""",unsafe_allow_html=True)
        st.markdown(f'<div class="sec-title">🩺 Tick All Applicable Comorbid Conditions</div>',unsafe_allow_html=True)
        st.caption("Each condition adds weighted points to the CCI score.")
        items=list(CCI_CONDITIONS.items()); half=len(items)//2+len(items)%2
        cc1,cc2=st.columns(2,gap="large")
        for i,(cond,pts) in enumerate(items):
            with (cc1 if i<half else cc2): st.checkbox(f"{cond}  (+{pts})",key=f"cci_{cond}")
        sel3=[(c,p) for c,p in CCI_CONDITIONS.items() if st.session_state.get(f"cci_{c}",False)]
        cp3=sum(p for _,p in sel3); ct3=ca2+cp3
        st.markdown(f"""<div style="margin-top:20px;background:{T['chip_bg']};border:1px solid {T['teal']};
            border-radius:12px;padding:16px 22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
          <span style="font-size:13px;color:{T['text_secondary']};">Calculated CCI:</span>
          <span style="font-family:'Space Grotesk',sans-serif;font-size:34px;font-weight:700;color:{T['teal']};">{ct3}</span>
          <span style="font-size:13px;color:{T['text_secondary']};">= Age pts ({ca2}) + Condition pts ({cp3})</span>
        </div>""",unsafe_allow_html=True)

    with tab3:
        st.markdown("<br>",unsafe_allow_html=True)
        age_r=st.session_state.get("age_input"); ca_r=st.session_state.get("ca_input","Not selected")
        mv_r=st.session_state.get("mv_input","Not selected"); inr_r=st.session_state.get("inr_input")
        plt_r=st.session_state.get("plt_input"); hct_r=st.session_state.get("hct_input")
        cl_r=st.session_state.get("cl_input"); cfs_r=st.session_state.get("cfs_input","— Select CFS score —")
        cci_age_r=cci_age(age_r)
        cci_cond_r=sum(p for c,p in CCI_CONDITIONS.items() if st.session_state.get(f"cci_{c}",False))
        cci_r=cci_age_r+cci_cond_r
        errs=[]
        if age_r is None: errs.append("Age")
        if ca_r=="Not selected": errs.append("Cardiac arrest status")
        if mv_r=="Not selected": errs.append("Mechanical ventilation")
        if inr_r is None: errs.append("INR")
        if plt_r is None: errs.append("Platelet count")
        if hct_r is None: errs.append("Hematocrit")
        if cl_r is None: errs.append("Chloride")
        if cfs_r=="— Select CFS score —": errs.append("Clinical Frailty Scale")
        if errs:
            st.markdown(f"""<div style="background:{T['err_bg']};border:1px solid {T['err_border']};
                border-radius:12px;padding:20px 24px;">
              <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;
                          color:#f43f5e;margin-bottom:10px;">⚠️ Incomplete Patient Data</div>
              <div style="font-size:13px;color:{T['text_secondary']};line-height:1.8;">
                Complete the following in the <strong style="color:{T['text_primary']};">Patient Details</strong> tab:<br>
                <strong style="color:{T['text_primary']};">{" · ".join(errs)}</strong></div>
            </div>""",unsafe_allow_html=True)
            st.stop()
        ca_n=1 if ca_r=="Yes" else 0; mv_n=1 if mv_r=="Yes" else 0
        cfs_n=int(cfs_r.split("—")[0].strip())
        comp,exp_s,bd=compute(age_r,ca_n,inr_r,mv_n,cci_r,cfs_n,hct_r,cl_r)
        m7=mort_at(exp_s,7); m14=mort_at(exp_s,14); m30=mort_at(exp_s,30)
        rl=classify(m30); rc=RISK_CFG[rl]
        emoji={"LOW":"🟢","MODERATE":"🟡","HIGH":"🔴","VERY HIGH":"🟣"}[rl]
        st.markdown(f"""<div class="risk-banner" style="background:{rc['bg']};border-color:{rc['border']};color:{rc['color']};">
          <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
            <div style="font-size:52px;line-height:1;flex-shrink:0;">{emoji}</div>
            <div style="flex:1;min-width:220px;">
              <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;opacity:0.75;margin-bottom:6px;">
                30-Day Mortality Risk Classification</div>
              <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px;">
                <span style="font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;">{rl}</span>
                <span style="background:{rc['color']};color:#fff;border-radius:20px;padding:4px 16px;font-size:12px;font-weight:700;">{rc['band']}</span>
              </div>
              <div style="font-size:13px;opacity:0.92;line-height:1.65;">
                <strong>Clinical Recommendation:</strong> {rc['rec']}</div>
            </div>
            <div style="text-align:right;flex-shrink:0;">
              <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;opacity:0.7;">30-Day Risk</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:54px;font-weight:700;line-height:1;">
                {m30:.1f}<span style="font-size:22px;">%</span></div>
            </div>
          </div>
        </div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        mc=st.columns(4,gap="small")
        for col,(lbl,val,clr,sub) in zip(mc,[
            ("Composite Score",f"{comp:.4f}",T['teal'],"Cox PH model"),
            ("7-Day Mortality",f"{m7:.2f}%",RISK_CFG[classify(m7)]["color"],classify(m7)),
            ("14-Day Mortality",f"{m14:.2f}%",RISK_CFG[classify(m14)]["color"],classify(m14)),
            ("30-Day Mortality",f"{m30:.2f}%",rc["color"],rl)]):
            with col:
                st.markdown(f"""<div class="metric-chip" style="border-color:{clr}44;">
                  <div class="mc-label">{lbl}</div><div class="mc-val" style="color:{clr};">{val}</div>
                  <div class="mc-sub" style="color:{clr};">{sub}</div></div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        cha,chb=st.columns([3,2],gap="large")
        pp,pb,pg,pf=T['plot_paper'],T['plot_bg'],T['plot_grid'],T['plot_font']
        with cha:
            st.markdown(f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:13px;font-weight:700;color:{T["text_primary"]};margin-bottom:10px;">📈 Mortality Risk Curve — Days 1 to 30</div>',unsafe_allow_html=True)
            dx=list(range(1,31)); dy=[mort_at(exp_s,d) for d in dx]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=dx,y=dy,mode="lines",line=dict(color=rc["color"],width=2.5),
                fill="tozeroy",fillcolor=rc["bg"].replace("0.09","0.18"),
                hovertemplate="Day %{x}: %{y:.2f}%<extra></extra>"))
            for kd,kn in [(7,"D7"),(14,"D14"),(30,"D30")]:
                kv=mort_at(exp_s,kd)
                fig.add_vline(x=kd,line_dash="dot",line_color="rgba(143,163,184,0.35)" if st.session_state.dark_mode else "rgba(77,106,135,0.30)",line_width=1)
                fig.add_annotation(x=kd,y=kv+max(dy)*0.09,text=f"<b>{kn}: {kv:.1f}%</b>",
                    showarrow=False,font=dict(size=11,color=rc["color"]))
            fig.update_layout(
                xaxis=dict(title="ICU Day",color=pf,gridcolor=pg,showgrid=True,linecolor=pg),
                yaxis=dict(title="Mortality Risk (%)",color=pf,gridcolor=pg,range=[0,max(max(dy)*1.3,5)]),
                height=295,margin=dict(l=40,r=20,t=10,b=40),paper_bgcolor=pp,plot_bgcolor=pb,
                showlegend=False,font=dict(family="DM Sans",color=pf))
            st.plotly_chart(fig,use_container_width=True)
        with chb:
            st.markdown(f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:13px;font-weight:700;color:{T["text_primary"]};margin-bottom:10px;">🧩 Factor Score Contributions</div>',unsafe_allow_html=True)
            sb=pd.DataFrame([{"F":k,"V":v,"C":"#f43f5e" if v>=0 else "#10b981"} for k,v in bd.items()]).sort_values("V",ascending=True)
            fig2=go.Figure(go.Bar(x=sb["V"],y=sb["F"],orientation="h",marker_color=sb["C"],
                text=sb["V"].apply(lambda x:f"{x:+.3f}"),textposition="outside",
                textfont=dict(size=10,family="JetBrains Mono",color=pf),
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
            fig2.update_layout(xaxis=dict(title="Contribution",color=pf,gridcolor=pg,linecolor=pg),
                yaxis=dict(color=pf),height=295,margin=dict(l=10,r=60,t=10,b=40),
                paper_bgcolor=pp,plot_bgcolor=pb,showlegend=False,font=dict(family="DM Sans",color=pf))
            st.plotly_chart(fig2,use_container_width=True)
        st.markdown(f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:13px;font-weight:700;color:{T["text_primary"]};margin:4px 0 12px 0;">🔬 Cox PH Score Breakdown</div>',unsafe_allow_html=True)
        bl=list(BETA.values())
        rows=[{"Factor":k,"β Coefficient":f"{bl[i]:+.5f}","Contribution":f"{v:+.6f}"} for i,(k,v) in enumerate(bd.items())]
        rows.append({"Factor":"▶  COMPOSITE SCORE (Σ)","β Coefficient":"—","Contribution":f"{comp:+.6f}"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        with st.expander("📋 Full Day-by-Day Survival & Mortality Table"):
            st.dataframe(pd.DataFrame([{"Day":d,"Survival (%)":f"{100-mort_at(exp_s,d):.4f}",
                "Mortality (%)":f"{mort_at(exp_s,d):.4f}","Band":classify(mort_at(exp_s,d))}
                for d in sorted(BASELINE_HAZARD)]),hide_index=True,use_container_width=True)
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("🔄  New Patient Assessment",type="secondary",key="new_pt"):
            keep={"page","dark_mode"}
            for k in list(st.session_state.keys()):
                if k not in keep: del st.session_state[k]
            st.rerun()
        st.markdown(f"""<div style="margin-top:22px;background:{T['warn_bg']};border:1px solid {T['warn_border']};
            border-radius:10px;padding:14px 18px;font-size:12px;color:{T['text_secondary']};line-height:1.7;">
          ⚠️ <strong style="color:{T['warn_text']};">Clinical Disclaimer:</strong>
          Decision-support aid for trained clinical professionals only. Does not replace multidisciplinary
          team judgement. All outputs are probabilistic estimates from a validated CPH model.
          <strong>Version 1.0 — September 2025.</strong></div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════ ABOUT ═══
elif page=="about":
    bk2,_=st.columns([1,9])
    with bk2:
        if st.button("← Home",key="bk_a"): st.session_state.page="home";st.rerun()
    st.markdown(f"""<div style="font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;
        color:{T['text_primary']};margin-bottom:4px;">About This Tool</div>
    <div style="font-size:14px;color:{T['text_secondary']};margin-bottom:26px;">
        Methodology, model variables, and clinical interpretation guide.</div>""",unsafe_allow_html=True)
    a1,a2=st.columns(2,gap="large")
    with a1:
        st.markdown(f"""<div class="icu-card">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      color:{T['teal']};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:14px;">
            ⚙️ Model Overview</div>
          <p style="font-size:14px;color:{T['text_secondary']};line-height:1.8;">
            Implements a <strong style="color:{T['text_primary']};">Cox Proportional Hazard (CPH) model</strong>
            validated for elderly ICU patients (≥ 65 years). Estimates mortality probability at specified
            time points using a composite score from clinical, laboratory and functional parameters.</p>
          <div style="background:{T['chip_bg']};border:1px solid {T['chip_border']};border-radius:8px;
                      padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:13px;color:{T['teal']};margin:14px 0;">
            S(t) = exp(−H₀(t) × exp(Σ βᵢXᵢ))</div>
          <p style="font-size:13px;color:{T['text_secondary']};line-height:1.7;">
            H₀(t) = baseline cumulative hazard · βᵢ = coefficients · Xᵢ = patient values</p>
        </div>""",unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div class="icu-card">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                      color:{T['teal']};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:14px;">
            📐 Variables & β Coefficients</div>""",unsafe_allow_html=True)
        for var,typ,beta,note in [
            ("Cardiac Arrest","Binary (0/1)","+2.7268","Strongest predictor"),
            ("Loge INR","ln(INR)","+1.7000","Coagulopathy marker"),
            ("Mechanical Ventilation","Binary (0/1)","+0.9076","Respiratory failure"),
            ("Loge Age","ln(Age)","+1.5523","Age-adjusted risk"),
            ("Loge CCI","ln(CCI)","+0.8597","Comorbidity burden"),
            ("CFS Score","1–8 scale","+0.0598","Frailty assessment"),
            ("Hematocrit","%","−0.0583","Anaemia / perfusion"),
            ("Chloride","mmol/L","−0.00096","Electrolyte balance"),
        ]:
            clr="#f43f5e" if "+" in beta else "#10b981"
            st.markdown(f"""<div class="coeff-row">
              <div style="flex:2;font-size:13px;color:{T['text_primary']};font-weight:500;">{var}</div>
              <div style="flex:1;font-size:11px;color:{T['text_secondary']};">{typ}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{clr};font-weight:700;min-width:74px;">{beta}</div>
              <div style="flex:2;font-size:11px;color:{T['text_secondary']};">{note}</div>
            </div>""",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(f"""<div class="icu-card">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;
                  color:{T['teal']};letter-spacing:0.09em;text-transform:uppercase;margin-bottom:18px;">
        🏥 Risk Band Clinical Guidance</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">""",unsafe_allow_html=True)
    for rl,cfg in RISK_CFG.items():
        st.markdown(f"""<div style="background:{cfg['bg']};border:1px solid {cfg['border']};border-radius:12px;padding:16px;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;font-weight:700;color:{cfg['color']};margin-bottom:4px;">{rl}</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;color:{cfg['color']};margin-bottom:8px;">{cfg['band']}</div>
          <div style="font-size:12px;color:{T['text_secondary']};line-height:1.6;">{cfg['rec']}</div>
        </div>""",unsafe_allow_html=True)
    st.markdown("</div></div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🔬  Start Patient Assessment",type="primary"):
        st.session_state.page="calculator";st.rerun()
