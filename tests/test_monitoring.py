"""
Unit Tests for Clinical Drift Monitoring Engine
AI-Assisted Pre-Hospital Patient Criticality Prediction System
"""

import pytest
import numpy as np
import pandas as pd
from src.monitoring import calculate_psi, ClinicalDriftMonitor
from src.generate_data import generate_synthetic_patient_data


def test_psi_calculation_identical_distributions():
    """Verify PSI is near zero for identical distributions."""
    np.random.seed(42)
    dist1 = np.random.normal(100, 15, size=2000)
    dist2 = np.random.normal(100, 15, size=2000)
    
    psi = calculate_psi(dist1, dist2)
    assert psi < 0.05, f"PSI for identical distributions should be < 0.05, got {psi}"


def test_psi_calculation_drifted_distribution():
    """Verify PSI is >= 0.20 when mean shifts substantially."""
    np.random.seed(42)
    dist1 = np.random.normal(100, 15, size=2000)
    dist2 = np.random.normal(135, 15, size=2000)  # Significant shift
    
    psi = calculate_psi(dist1, dist2)
    assert psi >= 0.20, f"PSI for shifted distribution should be >= 0.20, got {psi}"


def test_clinical_drift_monitor_stable_on_similar_splits():
    """Verify drift monitor reports STABLE on independent identically distributed samples."""
    train_df = generate_synthetic_patient_data(n_samples=1000, random_seed=10)
    test_df = generate_synthetic_patient_data(n_samples=1000, random_seed=20)
    
    monitor = ClinicalDriftMonitor(train_df)
    report = monitor.audit_feature_drift(test_df)
    
    assert report["overall_status"] in ["STABLE", "WARNING"]
    assert report["drifted_feature_count"] <= 1


def test_clinical_drift_monitor_detects_induced_feature_drift():
    """Verify drift monitor flags induced distribution shift in vitals."""
    train_df = generate_synthetic_patient_data(n_samples=500, random_seed=10)
    drifted_df = generate_synthetic_patient_data(n_samples=500, random_seed=20)
    
    # Induce massive shift in SpO2 and Heart Rate
    drifted_df["spo2"] = drifted_df["spo2"] - 20.0
    drifted_df["heart_rate"] = drifted_df["heart_rate"] + 45.0
    
    monitor = ClinicalDriftMonitor(train_df)
    report = monitor.audit_feature_drift(drifted_df)
    
    assert report["overall_status"] == "ACTION_REQUIRED"
    assert "spo2" in report["drifted_features"]
    assert "heart_rate" in report["drifted_features"]
