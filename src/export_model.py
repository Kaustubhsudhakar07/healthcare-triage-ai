"""
Edge & Cross-Platform Model Exporter
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Exports trained XGBoost model to Universal Binary JSON (UBJ) and JSON formats
for mobile ambulance tablets, embedded devices, and C++/Rust edge runtimes.
Benchmarks latency and verifies mathematical prediction parity.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import (
    ClinicalFeatureEngineer,
    build_preprocessor,
    TARGET_CONTINUOUS,
    TARGET_CATEGORICAL,
    ID_COLUMN
)


def export_edge_models(
    pipeline_path: str = "models/pipeline.joblib",
    output_dir: str = "models",
    test_path: str = "data/processed/test.csv"
) -> Dict[str, Any]:
    """
    Exports the trained estimator to universal edge formats and benchmarks inference latency.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading pipeline from '{pipeline_path}'...")
    pipeline = joblib.load(pipeline_path)
    estimator = pipeline.named_steps["regressor"]
    
    # 1. Export Native Universal JSON
    json_path = os.path.join(output_dir, "model.json")
    estimator.save_model(json_path)
    json_size_kb = os.path.getsize(json_path) / 1024.0
    print(f"  [+] Saved Universal JSON Model: {json_path} ({json_size_kb:.1f} KB)")
    
    # 2. Export Universal Binary JSON (UBJ - high speed binary format)
    ubj_path = os.path.join(output_dir, "model.ubj")
    estimator.save_model(ubj_path)
    ubj_size_kb = os.path.getsize(ubj_path) / 1024.0
    print(f"  [+] Saved Universal Binary Model: {ubj_path} ({ubj_size_kb:.1f} KB)")
    
    # 3. Parity and Latency Benchmarking
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        feature_cols = [c for c in test_df.columns if c not in [ID_COLUMN, TARGET_CONTINUOUS, TARGET_CATEGORICAL]]
        X_test = test_df[feature_cols]
        
        fe = pipeline.named_steps["feature_engineer"]
        preprocessor = pipeline.named_steps["preprocessor"]
        
        X_eng = fe.transform(X_test)
        X_proc = preprocessor.transform(X_eng)
        
        # Scikit-learn Pipeline latency
        t0 = time.perf_counter()
        n_iters = 500
        for _ in range(n_iters):
            _ = pipeline.predict(X_test[:1])
        sklearn_latency_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0
        
        # Reloaded Native XGBoost Model
        reloaded_xgb = xgb.XGBRegressor()
        reloaded_xgb.load_model(json_path)
        
        preds_original = estimator.predict(X_proc)
        preds_reloaded = reloaded_xgb.predict(X_proc)
        
        max_diff = float(np.max(np.abs(preds_original - preds_reloaded)))
        is_identical = max_diff < 1e-5
        
        print(f"\nModel Parity Verification: Max Difference = {max_diff:.2e} (Identical: {is_identical})")
        print(f"Single-Patient Inference Latency: {sklearn_latency_ms:.2f} ms")
        
        return {
            "json_path": json_path,
            "ubj_path": ubj_path,
            "json_size_kb": round(json_size_kb, 1),
            "ubj_size_kb": round(ubj_size_kb, 1),
            "max_difference": max_diff,
            "parity_verified": is_identical,
            "single_sample_latency_ms": round(sklearn_latency_ms, 2)
        }
    return {}


if __name__ == "__main__":
    export_edge_models()
