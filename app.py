"""
Step 8: Simple Streamlit demo app
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk Screener", page_icon="🩺", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    with open("best_model_name.txt") as f:
        model_name = f.read().strip()
    return model, scaler, feature_names, model_name

model, scaler, feature_names, model_name = load_artifacts()

st.title("🩺 Diabetes Risk Screener")
st.caption(f"Model: {model_name} · Trained on 253,680 CDC BRFSS 2015 survey records")

st.warning(
    "⚠️ **This is a screening/awareness tool only — NOT a medical diagnosis.** "
    "It estimates statistical risk based on population survey data. "
    "Please consult a doctor for an actual diagnosis or health concerns."
)

st.subheader("Tell us about yourself")

col1, col2 = st.columns(2)
with col1:
    high_bp = st.selectbox("Do you have high blood pressure?", ["No", "Yes"])
    high_chol = st.selectbox("Do you have high cholesterol?", ["No", "Yes"])
    chol_check = st.selectbox("Cholesterol checked in last 5 years?", ["Yes", "No"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    smoker = st.selectbox("Have you smoked 100+ cigarettes in your life?", ["No", "Yes"])
    stroke = st.selectbox("Ever had a stroke?", ["No", "Yes"])
    heart_disease = st.selectbox("Ever had coronary heart disease or heart attack?", ["No", "Yes"])
    phys_activity = st.selectbox("Physical activity in past 30 days (not job-related)?", ["Yes", "No"])
    fruits = st.selectbox("Eat fruit 1+ times/day?", ["Yes", "No"])
    veggies = st.selectbox("Eat vegetables 1+ times/day?", ["Yes", "No"])
    hvy_alcohol = st.selectbox("Heavy alcohol consumption?", ["No", "Yes"])

with col2:
    any_healthcare = st.selectbox("Do you have any healthcare coverage?", ["Yes", "No"])
    no_doc_cost = st.selectbox("Skipped doctor visit due to cost (past year)?", ["No", "Yes"])
    gen_health = st.select_slider("General health rating", options=[1,2,3,4,5],
                                    value=3, format_func=lambda x: {1:"Excellent",2:"Very Good",3:"Good",4:"Fair",5:"Poor"}[x])
    ment_health = st.slider("Days of poor mental health (past 30 days)", 0, 30, 0)
    phys_health = st.slider("Days of poor physical health (past 30 days)", 0, 30, 0)
    diff_walk = st.selectbox("Serious difficulty walking/climbing stairs?", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    age_band = st.select_slider("Age group", options=list(range(1,14)), value=7, format_func=lambda x: {
        1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",6:"45-49",7:"50-54",
        8:"55-59",9:"60-64",10:"65-69",11:"70-74",12:"75-79",13:"80+"}[x])
    education = st.select_slider("Education level", options=list(range(1,7)), value=4, format_func=lambda x: {
        1:"Never attended",2:"Elementary",3:"Some HS",4:"HS grad",5:"Some college",6:"College grad"}[x])
    income = st.select_slider("Income bracket", options=list(range(1,9)), value=5, format_func=lambda x: {
        1:"<$10k",2:"<$15k",3:"<$20k",4:"<$25k",5:"<$35k",6:"<$50k",7:"<$75k",8:"$75k+"}[x])

yn = lambda v: 1.0 if v == "Yes" else 0.0

input_dict = {
    "HighBP": yn(high_bp), "HighChol": yn(high_chol), "CholCheck": yn(chol_check),
    "BMI": bmi, "Smoker": yn(smoker), "Stroke": yn(stroke),
    "HeartDiseaseorAttack": yn(heart_disease), "PhysActivity": yn(phys_activity),
    "Fruits": yn(fruits), "Veggies": yn(veggies), "HvyAlcoholConsump": yn(hvy_alcohol),
    "AnyHealthcare": yn(any_healthcare), "NoDocbcCost": yn(no_doc_cost),
    "GenHlth": float(gen_health), "MentHlth": float(ment_health), "PhysHlth": float(phys_health),
    "DiffWalk": yn(diff_walk), "Sex": 1.0 if sex == "Male" else 0.0,
    "Age": float(age_band), "Education": float(education), "Income": float(income),
}
input_df = pd.DataFrame([input_dict])[feature_names]

if st.button("Check My Risk", type="primary"):
    proba = model.predict_proba(input_df)[0, 1]
    risk_pct = proba * 100

    if risk_pct < 20:
        level, color = "Low", "green"
    elif risk_pct < 50:
        level, color = "Moderate", "orange"
    else:
        level, color = "High", "red"

    st.markdown(f"### Estimated Risk: **:{color}[{risk_pct:.1f}% — {level}]**")
    st.progress(min(int(risk_pct), 100))

    st.subheader("What's driving this estimate?")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    contrib = pd.Series(shap_vals[0], index=feature_names).sort_values(key=abs, ascending=False).head(6)

    fig, ax = plt.subplots(figsize=(6,3.5))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in contrib.values]
    ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on risk (red = increases, green = decreases)")
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "🩺 **Remember:** This estimate is based on statistical patterns across a large population "
        "survey, not your individual medical history. If your risk is Moderate or High, or you have "
        "concerns, please talk to a doctor for proper testing (e.g. a fasting blood glucose or A1C test)."
    )

st.divider()
st.caption("Built for a social impact internship project · Data: CDC BRFSS 2015 Diabetes Health Indicators")
