from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_utils import FEATURE_COLUMNS
from src.recommendations import generate_recommendations, risk_category


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "diabetes_risk_model.joblib"


st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="D",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def collect_inputs() -> dict[str, float]:
    left, right = st.columns(2)

    with left:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose", min_value=40, max_value=250, value=120, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=140, value=72, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=5, max_value=80, value=25, step=1)

    with right:
        insulin = st.number_input("Insulin", min_value=10, max_value=900, value=90, step=1)
        bmi = st.number_input("BMI", min_value=12.0, max_value=60.0, value=28.0, step=0.1)
        pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.01, max_value=3.0, value=0.45, step=0.01)
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)

    return {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }


st.title("Multidataset Diabetes Health Risk Assessment")
st.caption("Educational ML project using PIMA/NIDDK diabetes data and synthetic realistic health records.")

artifact = load_model()

if artifact is None:
    st.warning("Model file not found. Run `python src/train_model.py` first.")
    st.stop()

model_name = artifact["model_name"]
pipeline = artifact["pipeline"]

st.sidebar.header("Model")
st.sidebar.write(f"Selected model: **{model_name}**")

with st.sidebar.expander("Model comparison"):
    metrics_df = pd.DataFrame(artifact["metrics"])
    st.dataframe(metrics_df.round(3), hide_index=True)

values = collect_inputs()

if st.button("Assess Risk", type="primary"):
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    probability = float(pipeline.predict_proba(input_df)[0][1])
    category = risk_category(probability)
    recommendations = generate_recommendations(values, probability)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Risk Category", category)
    metric_cols[1].metric("Diabetes Risk Probability", f"{probability * 100:.1f}%")
    metric_cols[2].metric("Model Used", model_name)

    st.subheader("Lifestyle Recommendations")
    for item in recommendations:
        st.write(f"- {item}")

    st.info("This system is for educational use only and does not replace professional medical diagnosis.")
