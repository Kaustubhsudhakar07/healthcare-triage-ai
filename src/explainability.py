"""
Explainable AI (XAI) and Local/Global SHAP Attribution Engine
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Computes global SHAP feature importance beeswarm/bar charts and generates
patient-specific local waterfall attributions and clinical narrative insights.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import (
    ClinicalFeatureEngineer,
    build_preprocessor,
    TARGET_CONTINUOUS,
    TARGET_CATEGORICAL,
    ID_COLUMN,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_BINARY_FEATURES
)


# Set clean plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def get_feature_names(preprocessor, X_sample: pd.DataFrame) -> List[str]:
    """
    Extracts output feature names from the fitted ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        num_cols = ALL_NUMERICAL_FEATURES
        cat_cols = []
        if hasattr(preprocessor.named_transformers_["cat"], "named_steps"):
            onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            if hasattr(onehot, "get_feature_names_out"):
                cat_cols = list(onehot.get_feature_names_out(CATEGORICAL_FEATURES))
        if not cat_cols:
            cat_cols = [f"sex_{v}" for v in ["Female", "Male", "Other"]]
        bin_cols = ALL_BINARY_FEATURES
        return num_cols + cat_cols + bin_cols


def clean_feature_name(name: str) -> str:
    """
    Converts raw encoded pipeline column names into clean, readable clinical labels.
    """
    clean = name.replace("num__", "").replace("bin__", "").replace("cat__", "").replace("remainder__", "")
    mapping = {
        "heart_rate": "Heart Rate (bpm)",
        "systolic_bp": "Systolic BP (mmHg)",
        "diastolic_bp": "Diastolic BP (mmHg)",
        "spo2": "SpO2 Oxygen (%)",
        "respiratory_rate": "Resp Rate (bpm)",
        "temperature": "Temperature (°C)",
        "gcs": "Glasgow Coma Scale",
        "pain_severity": "Pain Score (0-10)",
        "shock_index": "Shock Index (HR/SBP)",
        "pulse_pressure": "Pulse Pressure (mmHg)",
        "hypoxia_flag": "Severe Hypoxia Flag",
        "tachycardia_flag": "Tachycardia Flag",
        "coma_flag": "Coma / GCS<=8 Flag",
        "geriatric_risk": "Geriatric Age>=65",
        "ambulance_arrival": "Ambulance Transit",
        "walking_ability": "Ambulatory (Can Walk)",
        "altered_consciousness": "Altered Consciousness",
        "chest_pain": "Chest Pain",
        "difficulty_breathing": "Difficulty Breathing",
        "abdominal_pain": "Abdominal Pain",
        "injury_trauma": "Physical Trauma",
        "bleeding": "Active Bleeding",
        "fever": "Fever / Chills",
        "headache": "Severe Headache",
        "vomiting": "Vomiting",
        "oxygen_requirement": "Pre-Hospital O2 Given",
        "known_cardiac_history": "Cardiac History",
        "known_hypertension": "Known Hypertension",
        "known_diabetes": "Known Diabetes",
        "sex_Male": "Sex: Male",
        "sex_Female": "Sex: Female",
        "sex_Other": "Sex: Other",
        "age": "Patient Age"
    }
    return mapping.get(clean, clean.replace("_", " ").title())


class ClinicalExplainer:
    """
    Wrapper for SHAP explanations of pre-hospital triage predictions.
    """
    def __init__(self, full_pipeline, train_sample_df: pd.DataFrame):
        self.pipeline = full_pipeline
        self.fe = full_pipeline.named_steps["feature_engineer"]
        self.preprocessor = full_pipeline.named_steps["preprocessor"]
        self.regressor = full_pipeline.named_steps["regressor"]
        
        # Transform background sample
        X_eng = self.fe.transform(train_sample_df)
        self.X_bg_proc = self.preprocessor.transform(X_eng)
        self.feature_names = [clean_feature_name(fn) for fn in get_feature_names(self.preprocessor, X_eng)]
        
        # Initialize TreeExplainer or generic Explainer
        try:
            self.explainer = shap.TreeExplainer(self.regressor, self.X_bg_proc)
        except Exception:
            self.explainer = shap.Explainer(self.regressor.predict, self.X_bg_proc)

    def explain_instance(self, single_patient_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single patient record.
        """
        X_eng = self.fe.transform(single_patient_df)
        X_proc = self.preprocessor.transform(X_eng)
        
        shap_values = self.explainer(X_proc)
        
        # Extract values for the single row
        if len(shap_values.shape) == 1 or shap_values.values.ndim == 1:
            row_shap = shap_values.values
            base_val = shap_values.base_values
        else:
            row_shap = shap_values.values[0]
            base_val = shap_values.base_values[0] if hasattr(shap_values.base_values, "__len__") else shap_values.base_values
            
        factors = []
        for name, val in zip(self.feature_names, row_shap):
            factors.append({
                "feature": name,
                "shap_impact": round(float(val), 3),
                "abs_impact": abs(float(val))
            })
            
        factors = sorted(factors, key=lambda x: x["abs_impact"], reverse=True)
        
        # Generate clinical narrative
        narratives = []
        for f in factors[:5]:
            impact = f["shap_impact"]
            direction = "Elevates criticality by" if impact > 0 else "Reduces criticality by"
            sign = "+" if impact > 0 else "-"
            narratives.append(f"{sign}{abs(impact):.2f} pts: {f['feature']} ({direction} {abs(impact):.2f})")

        return {
            "base_value": round(float(base_val), 3),
            "top_factors": factors[:8],
            "narrative": narratives,
            "shap_values_raw": row_shap.tolist() if hasattr(row_shap, "tolist") else list(row_shap),
            "feature_names": self.feature_names
        }


def generate_global_shap_figures(
    pipeline,
    sample_df: pd.DataFrame,
    output_dir: str = "reports/figures",
    n_samples: int = 500
) -> None:
    """
    Computes and saves global SHAP Beeswarm and Bar summary figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fe = pipeline.named_steps["feature_engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    
    sample_subset = sample_df.sample(min(n_samples, len(sample_df)), random_state=42)
    X_eng = fe.transform(sample_subset)
    X_proc = preprocessor.transform(X_eng)
    
    feature_names = [clean_feature_name(fn) for fn in get_feature_names(preprocessor, X_eng)]
    
    explainer = shap.TreeExplainer(regressor)
    shap_vals = explainer(X_proc)
    shap_vals.feature_names = feature_names
    
    # 1. Beeswarm Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_vals, max_display=15, show=False)
    plt.title("Global SHAP Feature Attribution (Summary Beeswarm)", fontsize=13, fontweight="bold", pad=14)
    beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {beeswarm_path}")
    
    # 2. Mean Absolute SHAP Bar Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_vals, max_display=15, show=False)
    plt.title("Mean Absolute SHAP Feature Importance", fontsize=13, fontweight="bold", pad=14)
    bar_path = os.path.join(output_dir, "shap_feature_importance_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved: {bar_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts.")
    parser.add_argument("--test_path", type=str, default="data/processed/test.csv")
    parser.add_argument("--pipeline_path", type=str, default="models/pipeline.joblib")
    parser.add_argument("--output_fig_dir", type=str, default="reports/figures")
    
    args = parser.parse_args()
    
    print(f"Loading pipeline from '{args.pipeline_path}'...")
    pipeline = joblib.load(args.pipeline_path)
    
    print(f"Loading test data from '{args.test_path}'...")
    test_df = pd.read_csv(args.test_path)
    
    feature_cols = [c for c in test_df.columns if c not in [ID_COLUMN, TARGET_CONTINUOUS, TARGET_CATEGORICAL]]
    X_test = test_df[feature_cols]
    
    print("Generating Global SHAP figures...")
    generate_global_shap_figures(pipeline, X_test, output_dir=args.output_fig_dir)
    print("SHAP explainability figures generated successfully.")


if __name__ == "__main__":
    main()
