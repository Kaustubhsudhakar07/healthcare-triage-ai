"""
Unit Tests for Preprocessing Pipeline and Feature Engineering
AI-Assisted Pre-Hospital Patient Criticality Prediction System
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.generate_data import generate_synthetic_patient_data
from src.preprocessing import (
    ClinicalFeatureEngineer,
    build_preprocessor,
    build_full_pipeline,
    score_to_urgency_tier,
    clip_criticality_scores
)


def test_clinical_feature_engineer():
    """Verify calculation of Shock Index, Pulse Pressure, and Clinical Risk Flags."""
    df = generate_synthetic_patient_data(n_samples=50, random_seed=42)
    fe = ClinicalFeatureEngineer()
    df_eng = fe.fit_transform(df)
    
    assert "shock_index" in df_eng.columns
    assert "pulse_pressure" in df_eng.columns
    assert "hypoxia_flag" in df_eng.columns
    assert "tachycardia_flag" in df_eng.columns
    assert "coma_flag" in df_eng.columns
    assert "geriatric_risk" in df_eng.columns
    
    # Check shock index math
    expected_si = np.round(df["heart_rate"] / df["systolic_bp"], 3)
    assert np.allclose(df_eng["shock_index"].values, expected_si.values, atol=1e-2)


def test_column_transformer_fitting_and_shapes():
    """Verify ColumnTransformer handles numerical, categorical, and binary features without error."""
    df = generate_synthetic_patient_data(n_samples=100, random_seed=42)
    fe = ClinicalFeatureEngineer()
    df_eng = fe.fit_transform(df)
    
    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(df_eng)
    
    assert isinstance(X_proc, np.ndarray)
    assert X_proc.shape[0] == 100
    assert X_proc.shape[1] > 20
    assert not np.isnan(X_proc).any(), "Transformed feature matrix must not contain NaNs."


def test_full_pipeline_fit_predict():
    """Verify full end-to-end pipeline handles raw input DataFrames."""
    df = generate_synthetic_patient_data(n_samples=150, random_seed=42)
    feature_cols = [c for c in df.columns if c not in ["patient_id", "criticality_score", "urgency_category"]]
    
    X = df[feature_cols]
    y = df["criticality_score"].values
    
    pipeline = build_full_pipeline(Ridge())
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X[:10])
    clipped = clip_criticality_scores(preds)
    tiers = score_to_urgency_tier(clipped)
    
    assert len(preds) == 10
    assert (clipped >= 1.0).all() and (clipped <= 10.0).all()
    assert len(tiers) == 10
