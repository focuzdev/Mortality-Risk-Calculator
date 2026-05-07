"""
Elderly ICU Mortality Risk Calculator – Version 1.0 (Sept 2025)
Streamlit Application | Cox Proportional Hazard Model
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    .stApp { background-color: #f0f4f8; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #0d2137 100%);
    }
    [data-testid="stSidebar"] * { color: #e0eaf4 !important; }
    [data-testid="stSidebar"] label { color: #b0cde8 !important; font-weight: 600; }

    .section-header {
        background: linear-gradient(90deg, #1a3a5c, #2d6a9f);
        color: white !important;
        padding: 9px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 14px 0 6px 0;
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2d6a9f;
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 12px; color: #6b7c93; font-weight: 600; text-transform: uppercase; }
    .metric-card p  { margin: 4px 0 0 0; font-size: 28px; font-weight: 800; color: #1a3a5c; }

    .rec-box {
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
        font-size: 14px;
        line-height: 1.7;
    }

    .waiting-box {
        background: white;
        border-radius: 12px;
        padding: 50px 30px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        color: #8a9bb0;
    }

    .disclaimer {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 12px;
        color: #5d4037;
        margin-top: 18px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL CONSTANTS  (extracted from Excel v1.0)
# ─────────────────────────────────────────────
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
    1:  0.000007, 2:  0.000015, 3:  0.000019, 4:  0.000038,
    5:  0.000082, 6:  0.000099, 7:  0.000109, 8:  0.000132,
    9:  0.000145, 10: 0.000159, 11: 0.000179, 12: 0.000200,
    13: 0.000200, 14: 0.000200, 15: 0.000248, 16: 0.000275,
    17: 0.000334, 18: 0.000367, 19: 0.000403, 20: 0.000442,
    21: 0.000442, 22: 0.000491, 24: 0.000561, 26: 0.000652,
    30: 0.000756,
}

CCI_CONDITIONS = {
    "Myocardial infarction":                       1,
    "Congestive heart failure":                    1,
    "Peripheral vascular disease":                 1,
    "Cerebrovascular accident / TIA":              1,
    "Dementia":                                    1,
    "Chronic pulmonary disease":                   1,
    "Connective tissue disease":                   1,
    "Peptic ulcer disease":                        1,
    "Mild liver disease":                          1,
    "Uncomplicated diabetes":                      1,
    "Hemiplegia":                                  2,
    "Moderate to severe chronic kidney disease":   2,
    "Diabetes with end-organ damage":              2,
    "Localized solid tumor":                       2,
    "Leukemia":                                    2,
    "Lymphoma":                                    2,
    "Moderate to severe liver disease":            3,
    "Metastatic solid tumor":                      6,
    "AIDS":                                        6,
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

RISK_CONFIG = {
    "LOW": {
        "band": "< 10%", "color": "#1e7e34", "bg": "#d4edda", "border": "#28a745", "icon": "🟢",
        "rec": "ICU admission strongly recommended; anticipate favourable outcome. Standard monitoring protocols apply.",
    },
    "MODERATE": {
        "band": "10–30%", "color": "#b45309", "bg": "#fff3cd", "border": "#fd7e14", "icon": "🟡",
        "rec": "ICU admission appropriate; ensure early rehabilitation and family updates. Consider daily goal-setting and functional recovery planning.",
    },
    "HIGH": {
        "band": "> 30%", "color": "#b91c1c", "bg": "#f8d7da", "border": "#dc3545", "icon": "🔴",
        "rec": "ICU admission to be discussed in triage rounds; initiate family counselling and advance care planning. Palliative care consultation advised.",
    },
    "VERY HIGH": {
        "band": "> 60%", "color": "#6f1d1b", "bg": "#f5c6cb", "border": "#6f1d1b", "icon": "🔴",
        "rec": "ICU admission to be considered case-by-case; strong recommendation for goals-of-care and palliative care involvement. Early family conference essential.",
    },
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def classify_risk(pct):
    if pct < 10:    return "LOW"
    elif pct <= 30: return "MODERATE"
    elif pct <= 60: return "HIGH"
    else:           return "VERY HIGH"


def get_mort_at(exp_score, target_day):
    if target_day in BASELINE_HAZARD:
        return (1 - np.exp(-BASELINE_HAZARD[target_day] * exp_score)) * 100
    sorted_days = sorted(BASELINE_HAZARD.keys())
    for i in range(len(sorted_days) - 1):
        d1, d2 = sorted_days[i], sorted_days[i + 1]
        if d1 <= target_day <= d2:
            alpha = (target_day - d1) / (d2 - d1)
            h = BASELINE_HAZARD[d1] + alpha * (BASELINE_HAZARD[d2] - BASELINE_HAZARD[d1])
            return (1 - np.exp(-h * exp_score)) * 100
    return (1 - np.exp(-BASELINE_HAZARD[30] * exp_score)) * 100


def compute_risk(age, cardiac_arrest, inr, mech_vent, cci, cfs, hematocrit, chloride):
    log_age = np.log(max(age, 1))
    log_inr = np.log(max(inr, 0.001))
    log_cci = np.log(max(cci, 0.001))

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

    score_breakdown = {
        "Cardiac Arrest":         BETA["cardiac_arrest"]  * cardiac_arrest,
        "Loge INR":               BETA["log_inr"]         * log_inr,
        "Mechanical Ventilation": BETA["mechanical_vent"] * mech_vent,
        "Loge Age":               BETA["log_age"]         * log_age,
        "Loge CCI":               BETA["log_cci"]         * log_cci,
        "Clinical Frailty Scale": BETA["cfs"]             * cfs,
        "Hematocrit":             BETA["hematocrit"]      * hematocrit,
        "Chloride":               BETA["chloride"]        * chloride,
    }
    return composite, exp_score, score_breakdown


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a3a5c 0%,#2d6a9f 100%);
            padding:22px 30px;border-radius:12px;margin-bottom:24px;">
  <h1 style="color:white;margin:0;font-size:24px;font-weight:800;">
    🏥 Elderly ICU Mortality Risk Calculator
  </h1>
  <p style="color:#b0cde8;margin:6px 0 0 0;font-size:13px;">
    Cox Proportional Hazard Model &nbsp;|&nbsp; Version 1.0 (September 2025)
    &nbsp;|&nbsp; Validated for patients ≥ 65 years
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — BLANK INPUT FORM
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Particulars")
    st.caption("Fill in all fields below, then press **Calculate**.")
    st.markdown("---")

    # ── Demographics ──
    st.markdown('<div class="section-header">🧑 Demographics</div>', unsafe_allow_html=True)
    age = st.number_input(
        "Age upon admission (≥ 65 years)",
        min_value=65, max_value=120,
        value=None, placeholder="Enter age...", step=1,
    )

    # ── Acute Factors ──
    st.markdown('<div class="section-header">⚡ Acute Clinical Factors</div>', unsafe_allow_html=True)
    cardiac_arrest_input = st.radio(
        "Cardiac arrest prior to ICU admission?",
        options=["— Select —", "Yes", "No"], index=0, horizontal=True,
    )
    mech_vent_input = st.radio(
        "Mechanically ventilated upon ICU admission?",
        options=["— Select —", "Yes", "No"], index=0, horizontal=True,
    )

    # ── Laboratory Values ──
    st.markdown('<div class="section-header">🧪 Laboratory Values</div>', unsafe_allow_html=True)
    inr = st.number_input(
        "International Normalised Ratio (INR)",
        min_value=0.5, max_value=20.0,
        value=None, placeholder="e.g. 1.2", step=0.1, format="%.1f",
    )
    platelet = st.number_input(
        "Platelet count (×10⁹/L)",
        min_value=1, max_value=1000,
        value=None, placeholder="e.g. 150", step=1,
    )
    hematocrit = st.number_input(
        "Hematocrit (%)",
        min_value=1.0, max_value=70.0,
        value=None, placeholder="e.g. 35.0", step=0.1, format="%.1f",
    )
    chloride = st.number_input(
        "Chloride (mmol/L)",
        min_value=60, max_value=140,
        value=None, placeholder="e.g. 102", step=1,
    )

    # ── Frailty ──
    st.markdown('<div class="section-header">🚶 Frailty Assessment</div>', unsafe_allow_html=True)
    cfs_options = ["— Select —"] + [CFS_LABELS[i] for i in range(1, 9)]
    cfs_input = st.selectbox("Clinical Frailty Scale (CFS)", options=cfs_options, index=0)

    # ── Charlson Comorbidity Index ──
    st.markdown('<div class="section-header">🩺 Charlson Comorbidity Index</div>', unsafe_allow_html=True)
    st.caption("Tick all conditions present in this patient:")
    selected_conditions = []
    for cond, pts in CCI_CONDITIONS.items():
        if st.checkbox(f"{cond}  (+{pts} pt{'s' if pts > 1 else ''})", key=f"cci_{cond}"):
            selected_conditions.append((cond, pts))

    # Auto-compute age contribution
    cci_age = 0
    if age is not None:
        if 65 <= age <= 69:   cci_age = 1
        elif 70 <= age <= 79: cci_age = 2
        elif age >= 80:       cci_age = 4

    cci_conditions_pts = sum(p for _, p in selected_conditions)
    cci_total = cci_age + cci_conditions_pts
    st.info(f"**Calculated CCI: {cci_total}**\nAge pts: {cci_age} | Condition pts: {cci_conditions_pts}")

    st.markdown("---")
    calculate = st.button("🔍  Calculate Mortality Risk", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# RESULTS AREA
# ─────────────────────────────────────────────
if not calculate:
    # ── Waiting state ──
    st.markdown("""
    <div class="waiting-box">
        <div style="font-size:60px;margin-bottom:16px;">📝</div>
        <h2 style="color:#2d6a9f;margin:0 0 10px 0;">Enter Patient Data to Begin</h2>
        <p style="font-size:15px;max-width:500px;margin:0 auto;line-height:1.8;color:#555;">
            Complete all fields in the <strong>sidebar on the left</strong> — including demographics,
            laboratory values, frailty score, and comorbidities.<br><br>
            Press <strong>🔍 Calculate Mortality Risk</strong> when ready.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📖 Risk Band Reference Guide")
    ref_data = [
        {"Risk Band": "🟢  LOW",       "30-Day Mortality": "< 10%",  "Clinical Action": "ICU admission strongly recommended; anticipate favourable outcome."},
        {"Risk Band": "🟡  MODERATE",  "30-Day Mortality": "10–30%", "Clinical Action": "ICU admission appropriate; ensure early rehabilitation and family updates."},
        {"Risk Band": "🔴  HIGH",      "30-Day Mortality": "> 30%",  "Clinical Action": "Discuss in triage rounds; initiate family counselling and advance care planning."},
        {"Risk Band": "🔴  VERY HIGH", "30-Day Mortality": "> 60%",  "Clinical Action": "Case-by-case; strong recommendation for goals-of-care and palliative care."},
    ]
    st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)

else:
    # ── Validate all fields ──
    errors = []
    if age is None:                              errors.append("Age")
    if cardiac_arrest_input == "— Select —":     errors.append("Cardiac arrest status")
    if mech_vent_input == "— Select —":          errors.append("Mechanical ventilation status")
    if inr is None:                              errors.append("INR")
    if platelet is None:                         errors.append("Platelet count")
    if hematocrit is None:                       errors.append("Hematocrit")
    if chloride is None:                         errors.append("Chloride")
    if cfs_input == "— Select —":               errors.append("Clinical Frailty Scale")

    if errors:
        st.error(f"⚠️ Please complete the following required field(s): **{', '.join(errors)}**")
        st.stop()

    # ── Parse inputs ──
    ca_val  = 1 if cardiac_arrest_input == "Yes" else 0
    mv_val  = 1 if mech_vent_input == "Yes" else 0
    cfs_val = int(cfs_input.split("–")[0].strip())

    # ── Run model ──
    composite, exp_score, score_breakdown = compute_risk(
        age, ca_val, inr, mv_val, cci_total, cfs_val, hematocrit, chloride
    )

    mort_7d  = get_mort_at(exp_score, 7)
    mort_14d = get_mort_at(exp_score, 14)
    mort_30d = get_mort_at(exp_score, 30)

    risk_label = classify_risk(mort_30d)
    rc         = RISK_CONFIG[risk_label]

    # ════════════════════════════════════════
    # RESULT 1 — RISK BANNER
    # ════════════════════════════════════════
    st.markdown(f"""
    <div class="rec-box" style="background:{rc['bg']};border-left:6px solid {rc['border']};">
        <div style="font-size:20px;font-weight:900;color:{rc['color']};">
            {rc['icon']} &nbsp;Mortality Risk Classification:
            <span style="background:{rc['color']};color:white;padding:4px 18px;
                         border-radius:20px;font-size:17px;margin-left:10px;">
                {risk_label} &nbsp;({rc['band']})
            </span>
        </div>
        <div style="margin-top:12px;font-size:14px;color:#2d2d2d;">
            <strong>📋 Clinical Recommendation:</strong><br>
            {rc['rec']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # RESULT 2 — KEY METRIC CARDS
    # ════════════════════════════════════════
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Composite Score", f"{composite:.4f}", "#2d6a9f"),
        ("7-Day Mortality",  f"{mort_7d:.2f}%",  RISK_CONFIG[classify_risk(mort_7d)]["border"]),
        ("14-Day Mortality", f"{mort_14d:.2f}%", RISK_CONFIG[classify_risk(mort_14d)]["border"]),
        ("30-Day Mortality", f"{mort_30d:.2f}%", rc["border"]),
    ]
    for col, (title, val, color) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color};">
              <h3>{title}</h3>
              <p style="color:{color};">{val}</p>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # RESULT 3 — CHARTS
    # ════════════════════════════════════════
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### 📈 Mortality Risk Curve — Days 1 to 30")
        all_days   = list(range(1, 31))
        mort_curve = [get_mort_at(exp_score, d) for d in all_days]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=all_days, y=mort_curve, mode="lines",
            line=dict(color=rc["border"], width=3),
            fill="tozeroy", fillcolor="rgba(220,53,69,0.10)",
        ))
        for kday, kname in [(7, "Day 7"), (14, "Day 14"), (30, "Day 30")]:
            kval = get_mort_at(exp_score, kday)
            fig.add_vline(x=kday, line_dash="dot", line_color="#666", line_width=1.2)
            fig.add_annotation(
                x=kday, y=kval + max(mort_curve) * 0.08,
                text=f"<b>{kname}: {kval:.1f}%</b>",
                showarrow=False, font=dict(size=11, color="#333"),
            )
        fig.update_layout(
            xaxis_title="ICU Day", yaxis_title="Mortality Risk (%)",
            yaxis=dict(range=[0, max(max(mort_curve) * 1.3, 5)]),
            height=300, margin=dict(l=40, r=20, t=10, b=40),
            paper_bgcolor="white", plot_bgcolor="#f9fbfd", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🧩 Factor Score Contributions")
        sb_df = pd.DataFrame([
            {"Factor": k, "Value": v, "Color": "#dc3545" if v >= 0 else "#28a745"}
            for k, v in score_breakdown.items()
        ]).sort_values("Value", ascending=True)

        fig2 = go.Figure(go.Bar(
            x=sb_df["Value"], y=sb_df["Factor"], orientation="h",
            marker_color=sb_df["Color"],
            text=sb_df["Value"].apply(lambda x: f"{x:+.3f}"),
            textposition="outside",
        ))
        fig2.update_layout(
            xaxis_title="Score Contribution", height=300,
            margin=dict(l=10, r=60, t=10, b=40),
            paper_bgcolor="white", plot_bgcolor="#f9fbfd", showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ════════════════════════════════════════
    # RESULT 4 — SCORE BREAKDOWN TABLE
    # ════════════════════════════════════════
    st.markdown("#### 🔬 Cox PH Score Breakdown")
    beta_vals = list(BETA.values())
    rows = []
    for i, (k, v) in enumerate(score_breakdown.items()):
        rows.append({
            "Factor": k,
            "Beta (β)": f"{beta_vals[i]:+.5f}",
            "Contribution": f"{v:+.6f}",
        })
    rows.append({"Factor": "▶  SUM — Composite Score", "Beta (β)": "—", "Contribution": f"{composite:+.6f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ════════════════════════════════════════
    # RESULT 5 — FULL DAY-BY-DAY TABLE (collapsed)
    # ════════════════════════════════════════
    with st.expander("📋 View Full Survival / Mortality Table (All Time-Points)"):
        t_rows = []
        for d in sorted(BASELINE_HAZARD.keys()):
            m = get_mort_at(exp_score, d)
            s = 100 - m
            t_rows.append({
                "ICU Day": d,
                "Survival Probability": f"{s:.4f}%",
                "Mortality Risk": f"{m:.4f}%",
                "Risk Band": classify_risk(m),
            })
        st.dataframe(pd.DataFrame(t_rows), hide_index=True, use_container_width=True)

    # ════════════════════════════════════════
    # RESULT 6 — CCI BREAKDOWN (collapsed)
    # ════════════════════════════════════════
    with st.expander("🩺 View Charlson Comorbidity Index Breakdown"):
        cci_rows = [{"Condition": c, "Points": p} for c, p in selected_conditions]
        cci_rows.append({"Condition": f"Age contribution ({age} yrs)", "Points": cci_age})
        cci_rows.append({"Condition": "── TOTAL CCI SCORE ──", "Points": cci_total})
        st.dataframe(pd.DataFrame(cci_rows), hide_index=True, use_container_width=True)

    # ── New Patient Button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄  Clear & Enter New Patient", use_container_width=False):
        st.rerun()

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool is intended for use by trained clinical
        professionals as a decision-support aid only. It does not replace clinical judgement,
        multidisciplinary team assessment, or individualised patient care. All predictions are
        probabilistic estimates derived from a validated Cox Proportional Hazard model and must be
        interpreted alongside the full clinical picture.<br><br>
        <strong>Version 1.0 — September 2025</strong> &nbsp;|&nbsp;
        Validated for elderly patients (≥ 65 years) admitted to the ICU.
    </div>
    """, unsafe_allow_html=True)
