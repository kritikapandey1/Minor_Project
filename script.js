const model = {
  features: [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
  ],
  medians: [2, 114, 72, 25, 111, 29, 0.366, 41],
  means: [
    2.8490853658536586,
    115.45426829268293,
    72.59959349593495,
    25.641260162601625,
    119.51321138211382,
    29.598272357723573,
    0.44742835365853656,
    43.920223577235774,
  ],
  scales: [
    2.546137114158384,
    26.544829815083745,
    11.166980937092559,
    8.543668603634986,
    71.63821160557227,
    6.5557519167608795,
    0.31591098743257234,
    17.56038471480984,
  ],
  coefficients: [
    0.11081179626824329,
    0.8545231711945198,
    0.00752228722505411,
    -0.06256960609912215,
    -0.05602781840080144,
    0.4441522725787159,
    0.21937424410925527,
    0.6110658346206167,
  ],
  intercept: -0.1148101853515725,
};

const zeroAsMissing = new Set(["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]);

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function getRiskCategory(probability) {
  if (probability < 0.35) {
    return "Low Risk";
  }
  if (probability < 0.65) {
    return "Medium Risk";
  }
  return "High Risk";
}

function getCategoryColor(category) {
  if (category === "Low Risk") {
    return "#1f8a5b";
  }
  if (category === "Medium Risk") {
    return "#c58b16";
  }
  return "#c7443e";
}

function collectValues(form) {
  const values = {};

  model.features.forEach((feature) => {
    const input = form.querySelector(`[name="${feature}"]`);
    const number = input ? Number(input.value) : 0;
    values[feature] = Number.isFinite(number) ? number : 0;
  });

  return values;
}

function predictProbability(values) {
  let logit = model.intercept;

  model.features.forEach((feature, index) => {
    let value = values[feature];

    if (zeroAsMissing.has(feature) && value === 0) {
      value = model.medians[index];
    }

    const scaledValue = (value - model.means[index]) / model.scales[index];
    logit += scaledValue * model.coefficients[index];
  });

  return sigmoid(logit);
}

function generateRecommendations(values, probability) {
  const recommendations = [];
  const category = getRiskCategory(probability);

  if (category === "High Risk") {
    recommendations.push("Consult a doctor and consider confirmatory tests such as fasting blood glucose or HbA1c.");
  } else if (category === "Medium Risk") {
    recommendations.push("Schedule regular glucose monitoring and improve daily food and activity habits.");
  } else {
    recommendations.push("Maintain healthy habits and continue periodic diabetes screening.");
  }

  if (values.Glucose >= 140) {
    recommendations.push("Reduce sugary drinks, sweets, and refined carbohydrates; prefer high-fiber whole foods.");
  } else if (values.Glucose >= 100) {
    recommendations.push("Watch carbohydrate portions and choose low-glycemic foods more often.");
  }

  if (values.BMI >= 30) {
    recommendations.push("Aim for gradual weight reduction through portion control and at least 30 minutes of walking most days.");
  } else if (values.BMI >= 25) {
    recommendations.push("Focus on weight maintenance or modest weight loss with balanced meals and regular exercise.");
  }

  if (values.BloodPressure >= 130) {
    recommendations.push("Limit excess salt, monitor blood pressure, and include potassium-rich foods if medically suitable.");
  }

  if (values.Insulin >= 180) {
    recommendations.push("Discuss insulin resistance risk with a healthcare professional.");
  }

  if (values.Age >= 45) {
    recommendations.push("Because age is a risk factor, keep regular diabetes screening in your routine.");
  }

  recommendations.push("Include vegetables, lean protein, whole grains, adequate sleep, hydration, and stress management.");
  return recommendations;
}

function renderResult(values, probability) {
  const category = getRiskCategory(probability);
  const percent = Math.round(probability * 100);
  const color = getCategoryColor(category);
  const degrees = Math.round(probability * 360);

  document.getElementById("probabilityText").textContent = `${percent}%`;
  document.getElementById("riskCategory").textContent = category;
  document.getElementById("riskMessage").textContent =
    "This result is calculated from the trained Logistic Regression model using the same feature order and scaling as the Python pipeline.";

  const meter = document.querySelector(".meter");
  meter.style.background = `conic-gradient(${color} ${degrees}deg, #e8eeeb ${degrees}deg)`;

  const list = document.getElementById("recommendations");
  list.innerHTML = "";

  generateRecommendations(values, probability).forEach((item) => {
    const listItem = document.createElement("li");
    listItem.textContent = item;
    list.appendChild(listItem);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("riskForm");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const values = collectValues(form);
    const probability = predictProbability(values);
    renderResult(values, probability);
  });
});
