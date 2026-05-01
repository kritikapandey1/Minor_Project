from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    combine_datasets,
    generate_synthetic_dataset,
    load_pima_dataset,
    save_project_datasets,
)


ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def replace_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[ZERO_AS_MISSING_COLUMNS] = cleaned[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return cleaned


def make_pipeline(model) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, FEATURE_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, x_train, x_test, y_train, y_test) -> dict[str, float | str]:
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probabilities = pipeline.predict_proba(x_test)[:, 1]
    else:
        scores = pipeline.decision_function(x_test)
        probabilities = (scores - scores.min()) / (scores.max() - scores.min())

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    metrics["selection_score"] = (
        0.40 * metrics["f1"]
        + 0.30 * metrics["recall"]
        + 0.20 * metrics["roc_auc"]
        + 0.10 * metrics["accuracy"]
    )
    return metrics


def main() -> None:
    data_dir = ROOT / "data"
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    pima_df = load_pima_dataset(data_dir)
    synthetic_df = generate_synthetic_dataset()
    combined_df = combine_datasets(pima_df, synthetic_df)
    combined_df = replace_invalid_zeros(combined_df)
    save_project_datasets(data_dir, pima_df, synthetic_df, combined_df)

    x = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    pipelines = {}

    for name, model in models.items():
        pipeline = make_pipeline(model)
        metrics = evaluate_model(name, pipeline, x_train, x_test, y_train, y_test)
        results.append(metrics)
        pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("selection_score", ascending=False)
    results_df.to_csv(model_dir / "model_comparison.csv", index=False)

    best_name = str(results_df.iloc[0]["model"])
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(x, y)

    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "features": FEATURE_COLUMNS,
        "metrics": results_df.to_dict(orient="records"),
    }
    joblib.dump(artifact, model_dir / "diabetes_risk_model.joblib")

    print("Model comparison:")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {model_dir / 'diabetes_risk_model.joblib'}")


if __name__ == "__main__":
    main()
