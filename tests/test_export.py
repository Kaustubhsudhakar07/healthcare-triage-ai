"""
Unit Tests for Model Export and Edge Serialization Parity
AI-Assisted Pre-Hospital Patient Criticality Prediction System
"""

import os
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from src.export_model import export_edge_models


def test_export_edge_models_parity_and_files():
    """Verify JSON and UBJ edge models exist and match scikit-learn outputs exactly."""
    result = export_edge_models(
        pipeline_path="models/pipeline.joblib",
        output_dir="models",
        test_path="data/processed/test.csv"
    )
    
    assert os.path.exists(result["json_path"])
    assert os.path.exists(result["ubj_path"])
    assert result["parity_verified"] is True
    assert result["max_difference"] < 1e-4
