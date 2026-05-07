"""
Elderly ICU Mortality Risk Calculator – Version 1.0 (Sept 2025)
Streamlit Application | Cox Proportional Hazard Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Elderly ICU Mortality Risk Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #0d2137 100%);
    }
    [data-testid="stSidebar"] * { color: #e0eaf4 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label { color: #b0cde8 !important; font-weight: 600; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1a3a5c, #2d6a9f);
        color: white !important;
        padding: 10px 18px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 12px 0 8px 0;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 15px;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2d6a9f;
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 13px; color: #6b7c93; font-weight: 600; }
    .metric-card p  { margin: 4px 0 0 0; font-size: 26px; font-weight: 800; color: #1a3a5c; }

    /* Recommendation box */
    .rec-box {
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Score table */
    .score-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid #e8edf2;
        font-size: 13px;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 12px;
        color: #5d4037;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL CONSTANTS (from the Excel dataset)
# ─────────────────────────────────────────────
BETA = {
    "cardiac_arrest":   2.7268,
    "log_inr":          1.7000,   # implied from composite = log(INR)*beta; tuned from sheet
    "mechanical_vent":  0.9076,
    "log_age":          1.55229,
    "log_cci":          0.8597,
    "cfs":              0.05976,
    "hematocrit":      -0.05827,
    "chloride":        -0.00096,
}

# Baseline hazard at each time-point (from the Excel table)
BASELINE_HAZARD = {
    1:  0.000007,
    2:  0.000015,
    3:  0.000019,
    4:  0.000038,
    5:  0.000082,
    6:  0.000099,
    7:  0.000109,
    8:  0.000132,
    9:  0.000145,
    10: 0.000159,
    11: 0.000179,
    12: 0.0002,
    13: 0.0002,
    14: 0.0002,
    15: 0.000248,
    16: 0.000275,
    17: 0.000334,
    18: 0.000367,
    19: 0.000403,
    20: 0.000442,
    21: 0.000442,
    22: 0.000491,
    24: 0.000561,
    26: 0.000652,
    30: 0.000756,
}

# Charlson Comorbidity Index conditions and their weights
CCI_CONDITIONS = {
    "Myocardial infarction":                           1,
    "Congestive heart failure":                        1,
    "Peripheral vascular disease":                     1,
    "Cerebrovascular accident / TIA":                  1,
    "Dementia":                                        1,
    "Chronic pulmonary disease":                       1,
    "Connective tissue disease":                       1,
    "Peptic ulcer disease":                            1,
    "Mild liver disease":                              1,
    "Uncomplicated diabetes":                          1,
    "Hemiplegia":                                      2,
    "Moderate to severe chronic kidney disease":       2,
    "Diabetes with end-organ damage":                  2,
    "Localized solid tumor":                           2,
    "Leukemia":                                        2,
    "Lymphoma":                                        2,
    "Moderate to severe liver disease":                3,
    "Metastatic solid tumor":                          6,
    "AIDS":                                            6,
}

CFS_LABELS = {
    1: "1 – Very Fit",
    2: "2 – Well",
    3: "3 – Managing Well",
    4: "4 – Vulnerable",
    5: "5 – Mildly Frail",
    6: "6 – Moderately Frail",
    7: "7 – Severely Frail",
    8: "8 – Very Severely Frail",
}

# ─────────────────────────────────────────────
# RISK CLASSIFICATION
# ─────────────────────────────────────────────
def classify_risk(pct: float) -> dict:
    if pct < 10:
        return {
            "band": "< 10%", "label": "LOW", "color": "#28a745",
            "bg": "#d4edda", "border": "#28a745",
            "recommendation": (
                "ICU admission strongly recommended; anticipate favourable outcome. "
                "Standard monitoring protocols apply."
            )
        }
    elif pct <= 30:
        return {
            "band": "10–30%", "label": "MODERATE", "color": "#fd7e14",
            "bg": "#fff3cd", "border": "#fd7e14",
            "recommendation": (
                "ICU admission appropriate; ensure early rehabilitation and family updates. "
                "Consider daily goal-setting and functional recovery planning."
            )
        }
    elif pct <= 60:
        return {
            "band": "> 30%", "label": "HIGH", "color": "#dc3545",
            "bg": "#f8d7da", "border": "#dc3545",
            "recommendation": (
                "ICU admission to be discussed in triage rounds; initiate family counselling "
                "and advance care planning. Palliative care consultation advised."
            )
        }
    else:
        return {
            "band": "> 60%", "label": "VERY HIGH", "color": "#6f1d1b",
            "bg": "#f5c6cb", "border": "#6f1d1b",
            "recommendation": (
                "ICU admission to be considered case-by-case; strong recommendation for "
                "goals-of-care and palliative care involvement. Early family conference essential."
            )
        }

# ─────────────────────────────────────────────
# CORE CALCULATION
# ─────────────────────────────────────────────
def compute_risk(age, cardiac_arrest, inr, mech_vent, cci, platelet, cfs, hematocrit, chloride):
    """
    Compute Cox PH composite score and survival/mortality probabilities.
    S(t) = exp(-H0(t) * exp(composite_score))
    """
    # Log transforms (add tiny epsilon to avoid log(0))
    log_age  = np.log(max(age, 1))
    log_inr  = np.log(max(inr, 0.001))
    log_cci  = np.log(max(cci, 0.001))

    composite = (
        BETA["cardiac_arrest"]  * cardiac_arrest +
        BETA["log_inr"]         * log_inr +
        BETA["mechanical_vent"] * mech_vent +
        BETA["log_age"]         * log_age +
        BETA["log_cci"]         * log_cci +
        BETA["cfs"]             * cfs +
        BETA["hematocrit"]      * hematocrit +
        BETA["chloride"]        * chloride
    )

    exp_score = np.exp(composite)

    # Build survival curve at defined time points
    days  = sorted(BASELINE_HAZARD.keys())
    surv  = []
    mort  = []
    for d in days:
        h0   = BASELINE_HAZARD[d]
        s_t  = np.exp(-h0 * exp_score)
        surv.append(s_t)
        mort.append(1 - s_t)

    # Score breakdown
    score_breakdown = {
        "Cardiac Arrest (β=2.7268)":          BETA["cardiac_arrest"]  * cardiac_arrest,
        "Loge INR (β=1.70)":                  BETA["log_inr"]         * log_inr,
        "Mechanical Ventilation (β=0.9076)":  BETA["mechanical_vent"] * mech_vent,
        "Loge Age (β=1.5523)":                BETA["log_age"]         * log_age,
        "Loge CCI (β=0.8597)":                BETA["log_cci"]         * log_cci,
        "Clinical Frailty Scale (β=0.0598)":  BETA["cfs"]             * cfs,
        "Hematocrit (β=−0.0583)":             BETA["hematocrit"]      * hematocrit,
        "Chloride (β=−0.00096)":              BETA["chloride"]        * chloride,
    }

    return composite, exp_score, days, surv, mort, score_breakdown


# ─────────────────────────────────────────────
# SIDEBAR – PATIENT INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Patient Parameters")
    st.markdown("---")

    st.markdown('<div class="section-header">🧑‍⚕️ Demographics</div>', unsafe_allow_html=True)
    age = st.number_input("Age (≥65 years)", min_value=65, max_value=120, value=100, step=1)

    st.markdown('<div class="section-header">⚡ Acute Factors</div>', unsafe_allow_html=True)
    cardiac_arrest = st.selectbox("Cardiac arrest prior to ICU admission", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
    mech_vent = st.selectbox("Mechanically ventilated upon ICU admission", [0, 1],
                              format_func=lambda x: "Yes" if x else "No")

    st.markdown('<div class="section-header">🧪 Laboratory Values</div>', unsafe_allow_html=True)
    inr = st.number_input("International Normalised Ratio (INR)", min_value=0.5, max_value=20.0,
                           value=5.0, step=0.1, format="%.1f")
    platelet = st.number_input("Platelet count (×10⁹/L)", min_value=1, max_value=1000,
                                value=80, step=1)
    hematocrit = st.number_input("Hematocrit (%)", min_value=1.0, max_value=70.0,
                                  value=28.8, step=0.1, format="%.1f")
    chloride = st.number_input("Chloride (mmol/L)", min_value=60, max_value=140,
                                value=90, step=1)

    st.markdown('<div class="section-header">📊 Comorbidity & Frailty</div>', unsafe_allow_html=True)
    cfs = st.select_slider("Clinical Frailty Scale (CFS)",
                            options=list(range(1, 9)),
                            value=7,
                            format_func=lambda x: CFS_LABELS[x])

    st.markdown("**Charlson Comorbidity Index (CCI)**")
    st.caption("Select all conditions that apply:")
    cci_score_age = 0
    if 65 <= age <= 69: cci_score_age = 1
    elif 70 <= age <= 79: cci_score_age = 2
    elif age >= 80: cci_score_age = 4

    selected_conditions = []
    for cond, pts in CCI_CONDITIONS.items():
        if st.checkbox(f"{cond} (+{pts})", key=f"cci_{cond}"):
            selected_conditions.append((cond, pts))

    cci_conditions_score = sum(pts for _, pts in selected_conditions)
    cci = cci_score_age + cci_conditions_score

    st.markdown(f"**Calculated CCI: `{cci}`** (Age contribution: {cci_score_age}, Conditions: {cci_conditions_score})")

    st.markdown("---")
    calculate_btn = st.button("🔍 Calculate Mortality Risk", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
            padding: 22px 30px; border-radius: 12px; margin-bottom: 24px;">
    <h1 style="color: white; margin: 0; font-size: 26px; font-weight: 800;">
        🏥 Elderly ICU Mortality Risk Calculator
    </h1>
    <p style="color: #b0cde8; margin: 6px 0 0 0; font-size: 14px;">
        Cox Proportional Hazard Model &nbsp;|&nbsp; Version 1.0 (September 2025)
        &nbsp;|&nbsp; Validated for patients ≥65 years
    </p>
</div>
""", unsafe_allow_html=True)

# Always compute (or on button press)
if True:
    composite, exp_score, days, surv, mort, score_breakdown = compute_risk(
        age, cardiac_arrest, inr, mech_vent, cci, platelet, cfs, hematocrit, chloride
    )

    # ── Risk at key time-points ──
    def get_mort_at(target_day):
        if target_day in BASELINE_HAZARD:
            return (1 - np.exp(-BASELINE_HAZARD[target_day] * exp_score)) * 100
        # interpolate
        sorted_days = sorted(BASELINE_HAZARD.keys())
        for i in range(len(sorted_days) - 1):
            d1, d2 = sorted_days[i], sorted_days[i+1]
            if d1 <= target_day <= d2:
                h1, h2 = BASELINE_HAZARD[d1], BASELINE_HAZARD[d2]
                alpha = (target_day - d1) / (d2 - d1)
                h_interp = h1 + alpha * (h2 - h1)
                return (1 - np.exp(-h_interp * exp_score)) * 100
        return (1 - np.exp(-BASELINE_HAZARD[30] * exp_score)) * 100

    mort_7d  = get_mort_at(7)
    mort_14d = get_mort_at(14)
    mort_30d = get_mort_at(30)

    risk = classify_risk(mort_30d)

    # ── TOP METRICS ROW ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#2d6a9f;">
            <h3>COMPOSITE SCORE</h3>
            <p>{composite:.4f}</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#17a2b8;">
            <h3>7-DAY MORTALITY RISK</h3>
            <p>{mort_7d:.1f}%</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#fd7e14;">
            <h3>14-DAY MORTALITY RISK</h3>
            <p>{mort_14d:.1f}%</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:{risk['color']};">
            <h3>30-DAY MORTALITY RISK</h3>
            <p style="color:{risk['color']};">{mort_30d:.1f}%</p>
        </div>""", unsafe_allow_html=True)

    # ── RISK BAND & RECOMMENDATION ──
    st.markdown(f"""
    <div class="rec-box" style="background:{risk['bg']}; border-left: 5px solid {risk['border']};">
        <strong style="font-size:16px; color:{risk['color']};">
            ⚠️ Risk Classification:
            <span class="risk-badge" style="background:{risk['color']}; color:white;">
                {risk['label']} ({risk['band']})
            </span>
        </strong><br>
        <span style="color:#333; margin-top:6px; display:block;">
            📋 <strong>Clinical Recommendation:</strong> {risk['recommendation']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ──
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### 📈 Mortality Risk Curve Over ICU Stay (Days 1–30)")
        # Build a denser curve via interpolation
        all_days = list(range(1, 31))
        mort_curve = []
        for d in all_days:
            mort_curve.append(get_mort_at(d))

        fig_curve = go.Figure()
        # Fill area
        fig_curve.add_trace(go.Scatter(
            x=all_days, y=mort_curve,
            mode="lines", name="Mortality Risk (%)",
            line=dict(color="#dc3545", width=3),
            fill="tozeroy", fillcolor="rgba(220,53,69,0.12)"
        ))
        # Highlight key days
        for kd, kname, kcolor in [(7, "Day 7", "#fd7e14"), (14, "Day 14", "#6f42c1"), (30, "Day 30", "#dc3545")]:
            kv = get_mort_at(kd)
            fig_curve.add_vline(x=kd, line_dash="dot", line_color=kcolor, line_width=1.5)
            fig_curve.add_annotation(x=kd, y=kv + 1, text=f"<b>{kname}<br>{kv:.1f}%</b>",
                                     showarrow=False, font=dict(size=11, color=kcolor))

        fig_curve.update_layout(
            xaxis_title="ICU Day",
            yaxis_title="Mortality Risk (%)",
            yaxis=dict(range=[0, max(max(mort_curve) * 1.25, 5)], tickformat=".1f"),
            height=320,
            margin=dict(l=40, r=20, t=20, b=40),
            paper_bgcolor="white",
            plot_bgcolor="#f9fbfd",
            showlegend=False,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with col_b:
        st.markdown("#### 🧩 Score Contribution by Factor")
        sb_df = pd.DataFrame([
            {"Factor": k.split(" (")[0], "Contribution": v}
            for k, v in score_breakdown.items()
        ])
        sb_df["Color"] = sb_df["Contribution"].apply(
            lambda x: "#dc3545" if x > 0 else "#28a745"
        )
        sb_df = sb_df.sort_values("Contribution", ascending=True)

        fig_bar = go.Figure(go.Bar(
            x=sb_df["Contribution"],
            y=sb_df["Factor"],
            orientation="h",
            marker_color=sb_df["Color"],
            text=sb_df["Contribution"].apply(lambda x: f"{x:+.3f}"),
            textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis_title="Score Contribution",
            height=320,
            margin=dict(l=10, r=60, t=20, b=40),
            paper_bgcolor="white",
            plot_bgcolor="#f9fbfd",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── DETAILED SCORE BREAKDOWN TABLE ──
    st.markdown("#### 🔬 Detailed Cox PH Score Breakdown")
    breakdown_rows = []
    for factor, contrib in score_breakdown.items():
        raw_name = factor.split(" (β")[0]
        breakdown_rows.append({
            "Factor": raw_name,
            "Score Contribution": f"{contrib:+.6f}",
        })

    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_df.loc[len(breakdown_df)] = {
        "Factor": "🔢 SUM OF COMPOSITE SCORE",
        "Score Contribution": f"{composite:+.6f}",
    }

    st.dataframe(
        breakdown_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn("Factor", width="large"),
            "Score Contribution": st.column_config.TextColumn("Score Contribution", width="medium"),
        }
    )

    # ── SURVIVAL PROBABILITY TABLE ──
    st.markdown("#### 📋 Survival & Mortality Probability at Each Time-Point")
    table_data = []
    for d, s, m in zip(days, surv, mort):
        table_data.append({
            "ICU Day": d,
            "Survival Probability": f"{s*100:.4f}%",
            "Mortality Risk": f"{m*100:.4f}%",
            "Risk Band": classify_risk(m*100)["label"],
        })
    tdf = pd.DataFrame(table_data)
    st.dataframe(
        tdf, use_container_width=True, hide_index=True,
        column_config={
            "ICU Day": st.column_config.NumberColumn("ICU Day", width="small"),
            "Survival Probability": st.column_config.TextColumn("Survival Probability"),
            "Mortality Risk": st.column_config.TextColumn("Mortality Risk"),
            "Risk Band": st.column_config.TextColumn("Risk Band"),
        }
    )

    # ── CCI SUMMARY ──
    if selected_conditions:
        st.markdown("#### 🩺 Selected Charlson Comorbidity Conditions")
        cci_rows = [{"Condition": c, "Points": p} for c, p in selected_conditions]
        cci_rows.append({"Condition": f"Age contribution ({age} years)", "Points": cci_score_age})
        cci_rows.append({"Condition": "**TOTAL CCI SCORE**", "Points": cci})
        st.dataframe(pd.DataFrame(cci_rows), use_container_width=True, hide_index=True)

    # ── DISCLAIMER ──
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Clinical Disclaimer:</strong>
        This tool is intended for use by trained clinical professionals as a decision-support aid only.
        It does not replace clinical judgement, multidisciplinary team assessment, or individualised patient care.
        All predictions are probabilistic estimates derived from a validated Cox Proportional Hazard model.
        Mortality risk figures should be interpreted in conjunction with the full clinical picture.
        <br><br>
        <strong>Version 1.0 (September 2025)</strong> &nbsp;|&nbsp;
        Validated for elderly patients (≥65 years) admitted to the ICU.
    </div>
    """, unsafe_allow_html=True)
