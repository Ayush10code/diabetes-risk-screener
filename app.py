"""
Polished UI/UX version: custom design system, Plotly animated gauge,
card-based wizard steps, color-coded SHAP factor chips.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Risk Screener", page_icon="🩺", layout="centered")

# ============================================================
# DESIGN SYSTEM — custom CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Page background + spacing rhythm */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 720px;
    }

    /* Typography scale */
    h1 { font-size: 28px !important; font-weight: 700 !important; color: #123B4A; }
    h2 { font-size: 20px !important; font-weight: 600 !important; color: #123B4A; margin-top: 0.5rem; }
    h3 { font-size: 16px !important; font-weight: 600 !important; color: #1D5A6E; }
    p, li, label, .stMarkdown { font-size: 15px; color: #33424A; }
    .caption-text { font-size: 13px; color: #7A8B92; }

    /* Card container */
    .app-card {
        background: #FFFFFF;
        border: 1px solid #E3EAEC;
        border-radius: 16px;
        padding: 28px 28px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(18, 59, 74, 0.04);
    }

    .sub-card {
        background: #F7FAFB;
        border: 1px solid #E9EEF0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    /* Step indicator */
    .step-track {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 8px 0 28px 0;
    }
    .step-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
        position: relative;
    }
    .step-circle {
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px; font-weight: 600;
        border: 2px solid #D6E1E4;
        background: #FFFFFF;
        color: #9BB0B6;
        z-index: 2;
    }
    .step-circle.active {
        background: #1D8A99;
        border-color: #1D8A99;
        color: white;
    }
    .step-circle.done {
        background: #E1F3EF;
        border-color: #1D8A99;
        color: #1D8A99;
    }
    .step-label {
        font-size: 11px;
        color: #7A8B92;
        margin-top: 6px;
        text-align: center;
    }
    .step-label.active { color: #1D8A99; font-weight: 600; }
    .step-line {
        position: absolute;
        top: 16px; left: -50%; right: 50%;
        height: 2px;
        background: #D6E1E4;
        z-index: 1;
    }
    .step-line.done { background: #1D8A99; }

    /* Disclaimer box — calm, not alarming */
    .disclaimer-box {
        background: #F4F8F4;
        border: 1px solid #D7E6D7;
        border-left: 4px solid #6BA36B;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13.5px;
        color: #3C5A3C;
        margin: 16px 0;
    }

    /* Factor chips */
    .factor-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 14px;
    }
    .factor-chip.risk-up { background: #FDEDEC; border: 1px solid #F5C6C0; }
    .factor-chip.risk-down { background: #EAF6EE; border: 1px solid #C3E6CE; }
    .chip-bar-bg {
        flex: 1;
        background: #ffffff90;
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
    }
    .chip-bar-fill { height: 100%; border-radius: 6px; }
    .chip-bar-fill.up { background: #D9534F; }
    .chip-bar-fill.down { background: #4CAF7D; }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #D6E1E4;
        font-weight: 500;
        transition: all 0.15s ease;
        padding: 0.5rem 1.2rem;
    }
    div.stButton > button:hover {
        border-color: #1D8A99;
        color: #1D8A99;
        transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"] {
        background: #1D8A99;
        border-color: #1D8A99;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #176E7A;
        color: white;
    }

    /* Result headline */
    .risk-headline {
        text-align: center;
        margin: 4px 0 2px 0;
    }
    .risk-level-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    X_train_ref = joblib.load("X_train_reference.pkl")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return model, feature_names, X_train_ref, metadata

model, feature_names, X_train_ref, metadata = load_artifacts()
THRESHOLD = metadata["chosen_threshold"]

if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

STEPS = ["Basic Info", "Health History", "Lifestyle", "Results"]
STEP_ICONS = ["👤", "🏥", "🏃", "📊"]

def go_next():
    st.session_state.step = min(st.session_state.step + 1, len(STEPS) - 1)

def go_back():
    st.session_state.step = max(st.session_state.step - 1, 0)

def restart():
    st.session_state.step = 0
    st.session_state.answers = {}

# ============================================================
# Header
# ============================================================
st.markdown("# 🩺 Diabetes Risk Screener")
st.markdown(
    f'<p class="caption-text">Model: {metadata["model_name"]} · trained on 253,680 CDC BRFSS 2015 survey records</p>',
    unsafe_allow_html=True
)

# ---- Step indicator (numbered circles + connecting line) ----
step_html = '<div class="step-track">'
for i, (label, icon) in enumerate(zip(STEPS, STEP_ICONS)):
    if i < st.session_state.step:
        circle_class, line_class = "done", "done"
    elif i == st.session_state.step:
        circle_class, line_class = "active", "done" if i > 0 else ""
    else:
        circle_class, line_class = "", ""
    line_html = f'<div class="step-line {line_class}"></div>' if i > 0 else ""
    label_class = "active" if i == st.session_state.step else ""
    step_html += (
        f'<div class="step-node">'
        f'{line_html}'
        f'<div class="step-circle {circle_class}">{icon}</div>'
        f'<div class="step-label {label_class}">{label}</div>'
        f'</div>'
    )
step_html += '</div>'
st.markdown(step_html, unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer-box">⚠️ <b>Screening tool only — not a medical diagnosis.</b> '
    'Estimates statistical risk from population survey patterns. Please consult a doctor for a real assessment.</div>',
    unsafe_allow_html=True
)

a = st.session_state.answers

# ==================================================================
# STEP 0 — Basic Info
# ==================================================================
if st.session_state.step == 0:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### Tell us about yourself")
    a["sex"] = st.selectbox("Sex", ["Female", "Male"], index=["Female","Male"].index(a.get("sex","Female")))
    a["age_band"] = st.select_slider(
        "📅 Age group", options=list(range(1,14)), value=a.get("age_band", 7),
        format_func=lambda x: {1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",6:"45-49",
                                7:"50-54",8:"55-59",9:"60-64",10:"65-69",11:"70-74",12:"75-79",13:"80+"}[x])
    a["bmi"] = st.number_input("⚖️ BMI (Body Mass Index)", min_value=10.0, max_value=70.0,
                                 value=a.get("bmi", 25.0), step=0.1)
    st.caption("Not sure of your BMI? It's roughly weight(kg) ÷ height(m)².")
    a["education"] = st.select_slider(
        "🎓 Education level", options=list(range(1,7)), value=a.get("education", 4),
        format_func=lambda x: {1:"Never attended",2:"Elementary",3:"Some HS",4:"HS grad",
                                5:"Some college",6:"College grad"}[x])
    a["income"] = st.select_slider(
        "💰 Income bracket", options=list(range(1,9)), value=a.get("income", 5),
        format_func=lambda x: {1:"<$10k",2:"<$15k",3:"<$20k",4:"<$25k",5:"<$35k",6:"<$50k",7:"<$75k",8:"$75k+"}[x])
    st.caption("Income and education help the model account for known access-to-care patterns — see 'How this works' on the results page.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.button("Next →", on_click=go_next, type="primary")

# ==================================================================
# STEP 1 — Health History
# ==================================================================
elif st.session_state.step == 1:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### Health history")

    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    st.markdown("**Vitals**")
    a["high_bp"] = st.selectbox("🩸 Do you have high blood pressure?", ["No", "Yes"], index=["No","Yes"].index(a.get("high_bp","No")))
    a["high_chol"] = st.selectbox("🧪 Do you have high cholesterol?", ["No", "Yes"], index=["No","Yes"].index(a.get("high_chol","No")))
    a["chol_check"] = st.selectbox("📋 Cholesterol checked in last 5 years?", ["Yes", "No"], index=["Yes","No"].index(a.get("chol_check","Yes")))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    st.markdown("**Conditions**")
    a["stroke"] = st.selectbox("🧠 Ever had a stroke?", ["No", "Yes"], index=["No","Yes"].index(a.get("stroke","No")))
    a["heart_disease"] = st.selectbox("❤️ Ever had coronary heart disease or heart attack?", ["No", "Yes"], index=["No","Yes"].index(a.get("heart_disease","No")))
    a["diff_walk"] = st.selectbox("🚶 Serious difficulty walking/climbing stairs?", ["No", "Yes"], index=["No","Yes"].index(a.get("diff_walk","No")))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-card">', unsafe_allow_html=True)
    st.markdown("**How you've been feeling**")
    a["gen_health"] = st.select_slider(
        "🌡️ General health rating", options=[1,2,3,4,5], value=a.get("gen_health", 3),
        format_func=lambda x: {1:"Excellent",2:"Very Good",3:"Good",4:"Fair",5:"Poor"}[x])
    a["ment_health"] = st.slider("🧘 Days of poor mental health (past 30 days)", 0, 30, a.get("ment_health", 0))
    a["phys_health"] = st.slider("🩹 Days of poor physical health (past 30 days)", 0, 30, a.get("phys_health", 0))
    st.caption("'Poor physical health' means illness or injury days — not workout soreness.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.button("← Back", on_click=go_back)
    c2.button("Next →", on_click=go_next, type="primary")

# ==================================================================
# STEP 2 — Lifestyle
# ==================================================================
elif st.session_state.step == 2:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### Lifestyle")
    a["phys_activity"] = st.selectbox("🏋️ Physical activity in past 30 days (not job-related)?", ["Yes", "No"], index=["Yes","No"].index(a.get("phys_activity","Yes")))
    a["fruits"] = st.selectbox("🍎 Eat fruit 1+ times/day?", ["Yes", "No"], index=["Yes","No"].index(a.get("fruits","Yes")))
    a["veggies"] = st.selectbox("🥦 Eat vegetables 1+ times/day?", ["Yes", "No"], index=["Yes","No"].index(a.get("veggies","Yes")))
    a["smoker"] = st.selectbox("🚬 Smoked 100+ cigarettes in your life?", ["No", "Yes"], index=["No","Yes"].index(a.get("smoker","No")))
    a["hvy_alcohol"] = st.selectbox("🍷 Heavy alcohol consumption?", ["No", "Yes"], index=["No","Yes"].index(a.get("hvy_alcohol","No")))
    a["any_healthcare"] = st.selectbox("💳 Do you have any healthcare coverage?", ["Yes", "No"], index=["Yes","No"].index(a.get("any_healthcare","Yes")))
    a["no_doc_cost"] = st.selectbox("🚫 Skipped doctor visit due to cost (past year)?", ["No", "Yes"], index=["No","Yes"].index(a.get("no_doc_cost","No")))
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.button("← Back", on_click=go_back)
    c2.button("See my results →", on_click=go_next, type="primary")

# ==================================================================
# STEP 3 — Results
# ==================================================================
else:
    yn = lambda v: 1.0 if v == "Yes" else 0.0
    input_dict = {
        "HighBP": yn(a["high_bp"]), "HighChol": yn(a["high_chol"]), "CholCheck": yn(a["chol_check"]),
        "BMI": a["bmi"], "Smoker": yn(a["smoker"]), "Stroke": yn(a["stroke"]),
        "HeartDiseaseorAttack": yn(a["heart_disease"]), "PhysActivity": yn(a["phys_activity"]),
        "Fruits": yn(a["fruits"]), "Veggies": yn(a["veggies"]), "HvyAlcoholConsump": yn(a["hvy_alcohol"]),
        "AnyHealthcare": yn(a["any_healthcare"]), "NoDocbcCost": yn(a["no_doc_cost"]),
        "GenHlth": float(a["gen_health"]), "MentHlth": float(a["ment_health"]), "PhysHlth": float(a["phys_health"]),
        "DiffWalk": yn(a["diff_walk"]), "Sex": 1.0 if a["sex"] == "Male" else 0.0,
        "Age": float(a["age_band"]), "Education": float(a["education"]), "Income": float(a["income"]),
    }
    input_df = pd.DataFrame([input_dict])[feature_names]

    placeholder = st.empty()
    with placeholder.container():
        st.markdown('<div class="app-card" style="text-align:center; padding: 48px 20px;">', unsafe_allow_html=True)
        st.markdown("🔬 **Analyzing your health profile against 253,680 records...**")
        bar = st.progress(0)
        for pct in range(0, 101, 20):
            bar.progress(pct)
            time.sleep(0.08)
        st.markdown('</div>', unsafe_allow_html=True)
    proba = model.predict_proba(input_df)[0, 1]
    placeholder.empty()

    risk_pct = proba * 100

    if risk_pct < 15:
        level, color, icon = "Low", "#1D9E75", "✅"
    elif risk_pct < 35:
        level, color, icon = "Moderate", "#EF9F27", "⚠️"
    else:
        level, color, icon = "High", "#D9534F", "🔺"

    # ---- Plotly animated gauge ----
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#B9C6CA"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 15], "color": "#E1F3EF"},
                {"range": [15, 35], "color": "#FDF1DC"},
                {"range": [35, 100], "color": "#FBE7E5"},
            ],
        },
    ))
    gauge_fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div class="risk-headline">'
        f'<span class="risk-level-badge" style="background:{color}22; color:{color};">{icon} {level} risk</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ---- Percentile comparison ----
    age_peers = X_train_ref[X_train_ref["Age"] == a["age_band"]]
    if len(age_peers) > 20:
        peer_proba = model.predict_proba(age_peers)[:, 1]
        percentile = (peer_proba < proba).mean() * 100
        st.markdown(
            f'<p style="text-align:center; color:#5B6D74; font-size:14px; margin-top:12px;">'
            f'📈 Higher than <b>{percentile:.0f}%</b> of people in your age group in this dataset.</p>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Narrative SHAP explanation as colored chips ----
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 What's driving this estimate")

    try:
        explainer = shap.TreeExplainer(model.calibrated_classifiers_[0].estimator)
    except Exception:
        explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    contrib = pd.Series(shap_vals[0], index=feature_names)

    friendly_names = {
        "HighBP": "High blood pressure", "HighChol": "High cholesterol", "BMI": "Your BMI",
        "GenHlth": "Your general health rating", "Age": "Your age group", "PhysActivity": "Physical activity level",
        "DiffWalk": "Difficulty walking", "HeartDiseaseorAttack": "Heart disease/attack history",
        "Stroke": "Stroke history", "Smoker": "Smoking history", "Income": "Income level",
        "Education": "Education level", "PhysHlth": "Recent physical health", "MentHlth": "Recent mental health",
        "Fruits": "Fruit intake", "Veggies": "Vegetable intake", "HvyAlcoholConsump": "Alcohol consumption",
        "CholCheck": "Cholesterol check history", "AnyHealthcare": "Healthcare coverage",
        "NoDocbcCost": "Cost-related care access", "Sex": "Sex",
    }
    icon_map = {
        "HighBP": "🩸", "HighChol": "🧪", "BMI": "⚖️", "GenHlth": "🌡️", "Age": "📅",
        "PhysActivity": "🏋️", "DiffWalk": "🚶", "HeartDiseaseorAttack": "❤️", "Stroke": "🧠",
        "Smoker": "🚬", "Income": "💰", "Education": "🎓", "PhysHlth": "🩹", "MentHlth": "🧘",
        "Fruits": "🍎", "Veggies": "🥦", "HvyAlcoholConsump": "🍷", "CholCheck": "📋",
        "AnyHealthcare": "💳", "NoDocbcCost": "🚫", "Sex": "🧑",
    }

    increasing = contrib[contrib > 0].sort_values(ascending=False).head(3)
    decreasing = contrib[contrib < 0].sort_values().head(3)
    max_abs = max(contrib.abs().max(), 1e-6)

    if len(increasing) > 0:
        st.markdown("**🔺 Increasing your risk**")
        for feat, val in increasing.items():
            width = int(min(abs(val) / max_abs, 1.0) * 100)
            chip_html = (
                f'<div class="factor-chip risk-up">'
                f'<span>{icon_map.get(feat,"•")}</span>'
                f'<span style="flex:1.4; font-weight:500;">{friendly_names.get(feat, feat)}</span>'
                f'<div class="chip-bar-bg"><div class="chip-bar-fill up" style="width:{width}%;"></div></div>'
                f'</div>'
            )
            st.markdown(chip_html, unsafe_allow_html=True)

    if len(decreasing) > 0:
        st.markdown("**🔻 Reducing your risk**")
        for feat, val in decreasing.items():
            width = int(min(abs(val) / max_abs, 1.0) * 100)
            chip_html = (
                f'<div class="factor-chip risk-down">'
                f'<span>{icon_map.get(feat,"•")}</span>'
                f'<span style="flex:1.4; font-weight:500;">{friendly_names.get(feat, feat)}</span>'
                f'<div class="chip-bar-bg"><div class="chip-bar-fill down" style="width:{width}%;"></div></div>'
                f'</div>'
            )
            st.markdown(chip_html, unsafe_allow_html=True)

    suggestion_map = {
        "PhysActivity": "General health guidelines suggest regular physical activity supports healthy blood sugar regulation — this is educational context, not a personalized plan.",
        "BMI": "Body weight is one of several factors linked to diabetes risk in population studies; a doctor can help interpret what's healthy for your specific body.",
        "HighBP": "Blood pressure and diabetes risk are often linked in health research — regular monitoring is generally recommended.",
        "Fruits": "Dietary patterns including fruit and vegetable intake are commonly studied in relation to metabolic health.",
        "Veggies": "Dietary patterns including fruit and vegetable intake are commonly studied in relation to metabolic health.",
        "Smoker": "Smoking is associated with a range of metabolic and cardiovascular risks in public health research.",
    }
    if len(increasing) > 0 and increasing.index[0] in suggestion_map:
        st.info(f"💡 {suggestion_map[increasing.index[0]]}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- How this works ----
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    with st.expander("ℹ️  How this works — click to see the method behind your score"):
        st.markdown(f"""
        This tool uses a **{metadata['model_name']}** model trained on 253,680 real survey
        responses from the CDC's BRFSS 2015 health survey.

        - Probability outputs were **calibrated**, so a "30% risk" reflects roughly 30-in-100
          odds based on similar profiles in the data.
        - The decision threshold ({THRESHOLD:.2f}) was tuned to catch about **80% of true
          diabetes/prediabetes cases** (recall), since missing a real case matters more than a
          false alarm in a screening context.
        - Across 5-fold cross-validation, recall was stable at
          **{metadata['cv_recall_mean']*100:.1f}% ± {metadata['cv_recall_std']*100:.1f}%**.
        - The chips above use **SHAP values** — a method measuring how much each answer pushed
          your prediction up or down relative to the average.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer-box">🩺 This estimate reflects statistical patterns across a large '
        'population survey, not your personal medical history. If your risk is Moderate or High, '
        'consider talking to a doctor about a fasting glucose or A1C test.</div>',
        unsafe_allow_html=True
    )

    summary_text = f"""Diabetes Risk Screening Summary
================================
Estimated risk: {risk_pct:.1f}% ({level})
Model: {metadata['model_name']} (calibrated, threshold={THRESHOLD:.2f})

Key factors increasing risk: {', '.join([friendly_names.get(f, f) for f in increasing.index]) if len(increasing) else 'None identified'}
Key factors reducing risk: {', '.join([friendly_names.get(f, f) for f in decreasing.index]) if len(decreasing) else 'None identified'}

DISCLAIMER: This is a screening/awareness tool, not a medical diagnosis.
Please consult a doctor for an accurate assessment.
"""
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download results (.txt)", summary_text, file_name="diabetes_risk_summary.txt")
    c2.button("← Start over", on_click=restart)

st.divider()
st.markdown(
    '<p class="caption-text" style="text-align:center;">Built for a social impact internship project · '
    'Data: CDC BRFSS 2015 Diabetes Health Indicators</p>',
    unsafe_allow_html=True
)