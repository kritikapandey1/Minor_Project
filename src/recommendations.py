from __future__ import annotations

from typing import Mapping


def risk_category(probability: float) -> str:
    if probability < 0.35:
        return "Low Risk"
    if probability < 0.65:
        return "Medium Risk"
    return "High Risk"


def generate_recommendations(values: Mapping[str, float], probability: float) -> list[str]:
    recommendations: list[str] = []
    category = risk_category(probability)

    if category == "High Risk":
        recommendations.append("Consult a doctor and consider confirmatory tests such as fasting blood glucose or HbA1c.")
    elif category == "Medium Risk":
        recommendations.append("Schedule regular glucose monitoring and improve daily food and activity habits.")
    else:
        recommendations.append("Maintain healthy habits and continue periodic diabetes screening.")

    glucose = values.get("Glucose", 0)
    bmi = values.get("BMI", 0)
    blood_pressure = values.get("BloodPressure", 0)
    insulin = values.get("Insulin", 0)
    age = values.get("Age", 0)

    if glucose >= 140:
        recommendations.append("Reduce sugary drinks, sweets, and refined carbohydrates; prefer high-fiber whole foods.")
    elif glucose >= 100:
        recommendations.append("Watch carbohydrate portions and choose low-glycemic foods more often.")

    if bmi >= 30:
        recommendations.append("Aim for gradual weight reduction through portion control and at least 30 minutes of walking most days.")
    elif bmi >= 25:
        recommendations.append("Focus on weight maintenance or modest weight loss with balanced meals and regular exercise.")

    if blood_pressure >= 130:
        recommendations.append("Limit excess salt, monitor blood pressure, and include potassium-rich foods if medically suitable.")

    if insulin >= 180:
        recommendations.append("Discuss insulin resistance risk with a healthcare professional.")

    if age >= 45:
        recommendations.append("Because age is a risk factor, keep regular diabetes screening in your routine.")

    recommendations.append("Include vegetables, lean protein, whole grains, adequate sleep, hydration, and stress management.")

    return recommendations
