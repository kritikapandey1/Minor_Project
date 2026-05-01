# Multidataset Health Risk Assessment and Lifestyle Recommendation System for Diabetes

This minor project predicts diabetes risk using the PIMA/NIDDK diabetes dataset plus a synthetic realistic health dataset. It compares multiple ML models, selects the best model using test metrics, and provides lifestyle recommendations based on the predicted risk and user health values.

## Features

- Loads the PIMA Indians Diabetes dataset.
- Generates a synthetic realistic diabetes health dataset.
- Combines both datasets into one training dataset.
- Cleans invalid zero values in medical columns.
- Compares Logistic Regression, KNN, SVM, Decision Tree, Random Forest, and Gradient Boosting.
- Selects the best model using F1-score, recall, ROC-AUC, and accuracy.
- Saves the final model with preprocessing.
- Exports browser-friendly model parameters.
- Provides a static HTML, CSS, and JavaScript website for diabetes risk prediction and lifestyle recommendations.

## Dataset Notes

- PIMA Indians Diabetes Dataset: commonly associated with the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
- NIDDK information is used as medical background for lifestyle recommendation rules.
- Synthetic realistic data is generated locally to add variety for project demonstration.

Useful NIDDK reference pages:

- https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-type-2-diabetes
- https://www.niddk.nih.gov/health-information/diabetes/overview/diet-eating-physical-activity

## Project Structure

```text
.
|-- index.html
|-- script.js
|-- styles.css
|-- requirements.txt
|-- README.md
|-- data/
|   `-- generated after training
|-- models/
|   `-- generated after training
`-- src/
    |-- data_utils.py
    |-- recommendations.py
    `-- train_model.py
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models and save the best one:

```bash
python src/train_model.py
```

Open the website:

Double-click `index.html`, or run a simple local server and open `http://localhost:8000`.

```bash
python -m http.server 8000
```

## Input Features

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

## Model Selection

The training script compares several models and saves the best one according to a project-friendly score:

```text
0.40 * F1-score + 0.30 * Recall + 0.20 * ROC-AUC + 0.10 * Accuracy
```

Recall is weighted highly because failing to identify a high-risk diabetes case is more serious than a false warning.

## Disclaimer

This project is for educational use only. It is not a medical diagnosis tool. Users should consult qualified healthcare professionals for medical advice.
