"""
Clinical Preprocessing and Feature Engineering Pipelines
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Implements scikit-learn ColumnTransformer with domain clinical feature engineering,
robust scaling, categorical one-hot encoding, and threshold mapping.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Feature Definitions
RAW_NUMERICAL_FEATURES = [
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature",
    "gcs",
    "pain_severity"
]

ENGINEERED_NUMERICAL_FEATURES = [
    "shock_index",
    "pulse_pressure"
]

ALL_NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES

CATEGORICAL_FEATURES = ["sex"]

RAW_BINARY_FEATURES = [
    "ambulance_arrival",
    "walking_ability",
    "altered_consciousness",
    "chest_pain",
    "difficulty_breathing",
    "abdominal_pain",
    "injury_trauma",
    "bleeding",
    "fever",
    "headache",
    "vomiting",
    "oxygen_requirement",
    "known_cardiac_history",
    "known_hypertension",
    "known_diabetes"
]

ENGINEERED_BINARY_FEATURES = [
    "hypoxia_flag",
    "tachycardia_flag",
    "coma_flag",
    "geriatric_risk"
]

ALL_BINARY_FEATURES = RAW_BINARY_FEATURES + ENGINEERED_BINARY_FEATURES

TARGET_CONTINUOUS = "criticality_score"
TARGET_CATEGORICAL = "urgency_category"
ID_COLUMN = "patient_id"
URGENCY_ORDER = ["Low", "Moderate", "Elevated", "High", "Critical"]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for pre-hospital triage.
    Calculates physiological indices: Shock Index, Pulse Pressure, Hypoxia Flag,
    Tachycardia Flag, Coma Flag, and Geriatric Risk.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # Ensure working with DataFrame
        if not isinstance(X_out, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
            
        # 1. Shock Index: Heart Rate / Systolic Blood Pressure
        if "heart_rate" in X_out.columns and "systolic_bp" in X_out.columns:
            safe_sbp = np.maximum(X_out["systolic_bp"].fillna(120.0), 30.0)
            X_out["shock_index"] = np.round(X_out["heart_rate"] / safe_sbp, 3)
            
        # 2. Pulse Pressure: Systolic BP - Diastolic BP
        if "systolic_bp" in X_out.columns and "diastolic_bp" in X_out.columns:
            X_out["pulse_pressure"] = np.round(X_out["systolic_bp"] - X_out["diastolic_bp"], 1)
            
        # 3. Clinical High-Risk Flags
        if "spo2" in X_out.columns:
            X_out["hypoxia_flag"] = (X_out["spo2"] < 90.0).astype(int)
            
        if "heart_rate" in X_out.columns:
            X_out["tachycardia_flag"] = (X_out["heart_rate"] > 100.0).astype(int)
            
        if "gcs" in X_out.columns:
            X_out["coma_flag"] = (X_out["gcs"] <= 8).astype(int)
            
        if "age" in X_out.columns:
            X_out["geriatric_risk"] = (X_out["age"] >= 65).astype(int)
            
        return X_out


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a leakage-free ColumnTransformer for pre-hospital features.
    
    Returns
    -------
    ColumnTransformer
        Fittable scikit-learn transformer.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ALL_NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("bin", bin_pipeline, ALL_BINARY_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor


def build_full_pipeline(model_estimator) -> Pipeline:
    """
    Builds the complete end-to-end pipeline including ClinicalFeatureEngineer,
    ColumnTransformer preprocessor, and estimator.
    
    Parameters
    ----------
    model_estimator : BaseEstimator
        Scikit-learn compatible regression or classification estimator.
        
    Returns
    -------
    Pipeline
        Full fittable scikit-learn pipeline.
    """
    return Pipeline([
        ("feature_engineer", ClinicalFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("regressor", model_estimator)
    ])


def score_to_urgency_tier(scores: np.ndarray) -> np.ndarray:
    """
    Vectorized threshold mapping from continuous criticality score [1.0, 10.0]
    to the 5 operational urgency tiers.
    
    Parameters
    ----------
    scores : np.ndarray
        Array of predicted criticality scores.
        
    Returns
    -------
    np.ndarray
        Array of string category labels ('Low', 'Moderate', 'Elevated', 'High', 'Critical').
    """
    scores = np.asarray(scores)
    categories = np.empty(scores.shape, dtype=object)
    
    categories[scores < 2.5] = "Low"
    categories[(scores >= 2.5) & (scores < 4.5)] = "Moderate"
    categories[(scores >= 4.5) & (scores < 6.5)] = "Elevated"
    categories[(scores >= 6.5) & (scores < 8.5)] = "High"
    categories[scores >= 8.5] = "Critical"
    
    return categories


def clip_criticality_scores(scores: np.ndarray) -> np.ndarray:
    """
    Clamps predictions strictly to the valid physiological range [1.0, 10.0]
    and rounds to 1 decimal place.
    """
    return np.round(np.clip(np.asarray(scores), 1.0, 10.0), 1)
