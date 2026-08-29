"""
Unit Tests for Synthetic Data Generation and Schema Validation
AI-Assisted Pre-Hospital Patient Criticality Prediction System
"""

import pytest
import numpy as np
import pandas as pd
from src.generate_data import generate_synthetic_patient_data
from src.data_validation import validate_dataset, EXPECTED_COLUMNS, PHYSIOLOGICAL_BOUNDS


def test_generate_synthetic_patient_data_shape_and_columns():
    """Verify that dataset generation creates expected record count and column set."""
    n_records = 100
    df = generate_synthetic_patient_data(n_samples=n_records, random_seed=42)
    
    assert len(df) == n_records
    assert set(df.columns) == set(EXPECTED_COLUMNS)
    assert not df.isnull().any().any(), "Raw generated data should not contain nulls."


def test_physiological_range_bounds():
    """Verify generated data respects physiological range boundaries."""
    df = generate_synthetic_patient_data(n_samples=500, random_seed=42)
    
    for col, (lower, upper) in PHYSIOLOGICAL_BOUNDS.items():
        assert df[col].min() >= lower, f"{col} has values below {lower}: min={df[col].min()}"
        assert df[col].max() <= upper, f"{col} has values above {upper}: max={df[col].max()}"


def test_blood_pressure_physiological_invariant():
    """Verify SBP > DBP across all generated patient records."""
    df = generate_synthetic_patient_data(n_samples=500, random_seed=42)
    assert (df["systolic_bp"] > df["diastolic_bp"]).all(), "All patients must have SBP > DBP."


def test_target_score_and_tier_consistency():
    """Verify criticality_score maps deterministically to urgency_category."""
    df = generate_synthetic_patient_data(n_samples=500, random_seed=42)
    
    for _, row in df.iterrows():
        score = row["criticality_score"]
        tier = row["urgency_category"]
        if score < 2.5:
            assert tier == "Low"
        elif score < 4.5:
            assert tier == "Moderate"
        elif score < 6.5:
            assert tier == "Elevated"
        elif score < 8.5:
            assert tier == "High"
        else:
            assert tier == "Critical"


def test_validation_suite_raises_on_invalid_data():
    """Verify validation suite raises ValueError on corrupted data."""
    df = generate_synthetic_patient_data(n_samples=50, random_seed=42)
    
    # Inject impossible physiological value
    corrupt_df = df.copy()
    corrupt_df.loc[0, "spo2"] = 115.0  # Impossible SpO2
    
    with pytest.raises(ValueError, match="Physiological out-of-bounds"):
        validate_dataset(corrupt_df)
        
    # Inject SBP <= DBP
    corrupt_bp = df.copy()
    corrupt_bp.loc[0, "systolic_bp"] = 70.0
    corrupt_bp.loc[0, "diastolic_bp"] = 80.0
    with pytest.raises(ValueError, match="SBP <= DBP"):
        validate_dataset(corrupt_bp)
