from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN = "Outcome"

PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def load_pima_dataset(data_dir: Path) -> pd.DataFrame:
    """Load PIMA data from local cache or the public CSV URL."""
    data_dir.mkdir(parents=True, exist_ok=True)
    local_path = data_dir / "pima_diabetes.csv"

    if local_path.exists():
        return pd.read_csv(local_path)

    df = pd.read_csv(PIMA_URL, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])
    df.to_csv(local_path, index=False)
    return df


def generate_synthetic_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic diabetes-risk records for demonstration."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 81, n_samples)
    pregnancies = np.where(
        age < 25,
        rng.poisson(0.8, n_samples),
        rng.poisson(2.4, n_samples),
    )
    pregnancies = np.clip(pregnancies, 0, 15)

    bmi = rng.normal(27 + 0.06 * (age - 35), 5.8, n_samples)
    bmi = np.clip(bmi, 16, 52)

    pedigree = rng.gamma(shape=2.0, scale=0.22, size=n_samples)
    pedigree = np.clip(pedigree, 0.05, 2.5)

    glucose = rng.normal(92 + 0.55 * (age - 30) + 1.45 * (bmi - 25) + 10 * pedigree, 18, n_samples)
    glucose = np.clip(glucose, 55, 230)

    blood_pressure = rng.normal(68 + 0.18 * (age - 30) + 0.45 * (bmi - 25), 10, n_samples)
    blood_pressure = np.clip(blood_pressure, 45, 125)

    skin_thickness = rng.normal(20 + 0.7 * (bmi - 22), 7, n_samples)
    skin_thickness = np.clip(skin_thickness, 7, 65)

    insulin = rng.normal(70 + 1.5 * (glucose - 90) + 2.2 * (bmi - 25), 45, n_samples)
    insulin = np.clip(insulin, 15, 500)

    risk_score = (
        -9.0
        + 0.037 * glucose
        + 0.075 * bmi
        + 0.025 * age
        + 0.55 * pedigree
        + 0.012 * blood_pressure
        + 0.05 * pregnancies
    )
    probability = 1 / (1 + np.exp(-risk_score))
    outcome = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "Pregnancies": pregnancies.astype(int),
            "Glucose": np.round(glucose).astype(int),
            "BloodPressure": np.round(blood_pressure).astype(int),
            "SkinThickness": np.round(skin_thickness).astype(int),
            "Insulin": np.round(insulin).astype(int),
            "BMI": np.round(bmi, 1),
            "DiabetesPedigreeFunction": np.round(pedigree, 3),
            "Age": age.astype(int),
            "Outcome": outcome.astype(int),
        }
    )


def combine_datasets(pima_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([pima_df, synthetic_df], ignore_index=True)
    return combined[FEATURE_COLUMNS + [TARGET_COLUMN]]


def save_project_datasets(data_dir: Path, pima_df: pd.DataFrame, synthetic_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    pima_df.to_csv(data_dir / "pima_diabetes.csv", index=False)
    synthetic_df.to_csv(data_dir / "synthetic_diabetes.csv", index=False)
    combined_df.to_csv(data_dir / "combined_diabetes.csv", index=False)
