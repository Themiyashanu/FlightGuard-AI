from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Airplane_Crashes_and_Fatalities_Since_1908.csv"
FALLBACK_DATA_PATH = ROOT_DIR / "Airplane_Crashes_and_Fatalities_Since_1908.csv"
MODELS_DIR = ROOT_DIR / "models"


def _load_dataset() -> pd.DataFrame:
    data_path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else FALLBACK_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Put the CSV file in data/raw/ or project root."
        )
    return pd.read_csv(data_path)


def _categorize_aircraft(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).lower()
    if any(w in text for w in ["boeing", "airbus", "dc-", "md-", "embraer"]):
        return "Commercial"
    if any(w in text for w in ["fighter", "bomber", "military"]):
        return "Military"
    if any(w in text for w in ["helicopter", "chopper"]):
        return "Helicopter"
    if any(w in text for w in ["cessna", "piper", "beech"]):
        return "General Aviation"
    return "Other"


def _extract_country(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value)
    if "," in text:
        return text.split(",")[-1].strip()
    return text.strip() if text.strip() else "Unknown"


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = (
        data.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["year"] = data["date"].dt.year.fillna(data["date"].dt.year.median())
        data["month"] = data["date"].dt.month.fillna(0)
    else:
        data["year"] = 0
        data["month"] = 0

    data["fatalities"] = pd.to_numeric(data.get("fatalities", 0), errors="coerce").fillna(0)
    data["aboard"] = pd.to_numeric(data.get("aboard", 0), errors="coerce").fillna(0)

    data["high_risk"] = (data["fatalities"] > 10).astype(int)

    def _col_or_default(name: str, default: str = "Unknown") -> pd.Series:
        if name in data.columns:
            return data[name].fillna(default)
        return pd.Series([default] * len(data), index=data.index)

    data["operator"] = _col_or_default("operator")
    data["is_military"] = (
        data["operator"].str.contains("military|army|navy|air force", case=False, na=False).astype(int)
    )
    data["is_commercial"] = (
        data["operator"].str.contains("airlines|airways|air", case=False, na=False).astype(int)
    )

    data["type"] = _col_or_default("type")
    data["aircraft_category"] = data["type"].map(_categorize_aircraft)

    data["location"] = _col_or_default("location")
    data["country"] = data["location"].map(_extract_country)

    data["summary"] = _col_or_default("summary", "")
    data["summary_length"] = data["summary"].str.len()
    data["weather_mentioned"] = data["summary"].str.contains(
        "weather|storm|wind|fog|rain|snow", case=False, na=False
    ).astype(int)
    data["mechanical_failure"] = data["summary"].str.contains(
        "engine|mechanical|failure|malfunction", case=False, na=False
    ).astype(int)
    data["pilot_error"] = data["summary"].str.contains(
        "pilot|crew|human error|mistake", case=False, na=False
    ).astype(int)

    data["broad_phase"] = _col_or_default("phase")
    return data


def build_feature_table(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "year": data["year"],
            "month": data["month"],
            "aboard": data["aboard"],
            "summary_length": data["summary_length"],
            "is_military": data["is_military"],
            "is_commercial": data["is_commercial"],
            "weather_mentioned": data["weather_mentioned"],
            "mechanical_failure": data["mechanical_failure"],
            "pilot_error": data["pilot_error"],
            "aircraft_category": data["aircraft_category"],
            "country": data["country"],
            "broad_phase": data["broad_phase"],
        }
    )
    target = data["high_risk"]
    return features, target


def train_and_save_models() -> Dict[str, float]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = _load_dataset()
    cleaned_df = preprocess_data(raw_df)
    X, y = build_feature_table(cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = [
        "year",
        "month",
        "aboard",
        "summary_length",
        "is_military",
        "is_commercial",
        "weather_mentioned",
        "mechanical_failure",
        "pilot_error",
    ]
    categorical_features = ["aircraft_category", "country", "broad_phase"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=1500, random_state=42, class_weight="balanced"
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_auc = -np.inf
    best_pipeline = None

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", estimator)]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        accuracy = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        results[model_name] = {"accuracy": accuracy, "auc": auc}

        if auc > best_auc:
            best_auc = auc
            best_name = model_name
            best_pipeline = pipeline

    metrics = {
        "best_model": best_name,
        "best_accuracy": results[best_name]["accuracy"],
        "best_auc": results[best_name]["auc"],
        "all_models": results,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "total_rows": int(X.shape[0]),
    }

    joblib.dump(best_pipeline, MODELS_DIR / "aviation_risk_model.pkl")
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    cleaned_df.to_csv(ROOT_DIR / "data" / "processed" / "processed_aviation_data.csv", index=False)
    return metrics


def predict_risk(input_dict: Dict) -> Dict[str, float]:
    model_path = MODELS_DIR / "aviation_risk_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model file missing. Run `python src/train_model.py` first.")

    model = joblib.load(model_path)
    input_df = pd.DataFrame([input_dict])
    prob = float(model.predict_proba(input_df)[:, 1][0])
    label = "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low"
    return {"probability": prob, "risk_level": label}
