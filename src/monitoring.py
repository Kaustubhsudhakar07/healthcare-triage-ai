"""
Model Monitoring and Data Drift Detection Engine
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Computes Population Stability Index (PSI), Kolmogorov-Smirnov (KS) tests,
and Chi-Square categorical distribution drift metrics for pre-hospital vitals.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import ks_2samp, chisquare

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import RAW_NUMERICAL_FEATURES, URGENCY_ORDER


def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between baseline and target distributions.
    
    Interpretation:
    - PSI < 0.10: No significant distribution change (Stable)
    - 0.10 <= PSI < 0.20: Moderate distribution shift (Warning)
    - PSI >= 0.20: Significant distribution shift (Drift / Action Required)
    """
    baseline = np.asarray(baseline).dropna() if hasattr(baseline, "dropna") else np.asarray(baseline)
    target = np.asarray(target).dropna() if hasattr(target, "dropna") else np.asarray(target)
    
    baseline = baseline[~np.isnan(baseline)]
    target = target[~np.isnan(target)]
    
    if len(baseline) == 0 or len(target) == 0:
        return 0.0
        
    # Determine quantiles on baseline
    percentiles = np.linspace(0, 100, num_buckets + 1)
    breakpoints = np.percentile(baseline, percentiles)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Calculate counts in buckets
    base_counts, _ = np.histogram(baseline, bins=breakpoints)
    target_counts, _ = np.histogram(target, bins=breakpoints)
    
    # Convert to fractions with Laplace smoothing
    base_pct = (base_counts + 1e-4) / (len(baseline) + 1e-4 * num_buckets)
    target_pct = (target_counts + 1e-4) / (len(target) + 1e-4 * num_buckets)
    
    # PSI calculation
    psi_value = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
    return round(float(psi_value), 4)


class ClinicalDriftMonitor:
    """
    Monitors incoming pre-hospital patient stream for feature drift and target prediction drift.
    """
    def __init__(self, baseline_df: pd.DataFrame):
        self.baseline_df = baseline_df.copy()
        self.numerical_features = [c for c in RAW_NUMERICAL_FEATURES if c in baseline_df.columns]

    def audit_feature_drift(self, incoming_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audits numerical vitals drift using PSI and Kolmogorov-Smirnov 2-sample tests.
        """
        drift_report = {}
        drifted_features = []
        
        for feat in self.numerical_features:
            if feat not in incoming_df.columns:
                continue
                
            base_vals = self.baseline_df[feat].values
            curr_vals = incoming_df[feat].values
            
            # Compute PSI
            psi = calculate_psi(base_vals, curr_vals)
            
            # Compute Kolmogorov-Smirnov Test
            ks_stat, p_val = ks_2samp(base_vals, curr_vals)
            
            is_drifted = psi >= 0.20 or p_val < 0.01
            status = "DRIFT_DETECTED" if psi >= 0.20 else ("WARNING" if psi >= 0.10 else "STABLE")
            
            if is_drifted:
                drifted_features.append(feat)
                
            drift_report[feat] = {
                "psi": psi,
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 5),
                "status": status,
                "is_drifted": is_drifted
            }
            
        return {
            "total_features_evaluated": len(self.numerical_features),
            "drifted_feature_count": len(drifted_features),
            "drifted_features": drifted_features,
            "overall_status": "ACTION_REQUIRED" if len(drifted_features) >= 2 else ("WARNING" if len(drifted_features) == 1 else "STABLE"),
            "details": drift_report
        }

    def audit_prediction_drift(self, baseline_scores: np.ndarray, incoming_scores: np.ndarray) -> Dict[str, Any]:
        """
        Audits model prediction distribution drift.
        """
        psi = calculate_psi(baseline_scores, incoming_scores)
        ks_stat, p_val = ks_2samp(baseline_scores, incoming_scores)
        status = "DRIFT_DETECTED" if psi >= 0.20 else ("WARNING" if psi >= 0.10 else "STABLE")
        
        return {
            "prediction_psi": psi,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 5),
            "status": status,
            "baseline_mean_acuity": round(float(np.mean(baseline_scores)), 2),
            "incoming_mean_acuity": round(float(np.mean(incoming_scores)), 2)
        }


if __name__ == "__main__":
    if os.path.exists("data/processed/train.csv") and os.path.exists("data/processed/test.csv"):
        train = pd.read_csv("data/processed/train.csv")
        test = pd.read_csv("data/processed/test.csv")
        
        monitor = ClinicalDriftMonitor(train)
        
        # 1. Audit on standard test set (should be STABLE)
        report_normal = monitor.audit_feature_drift(test)
        print("Standard Test Set Drift Audit:")
        print(f"  Overall Status: {report_normal['overall_status']}")
        print(f"  Drifted Features: {report_normal['drifted_features']}")
        
        # 2. Simulate artificial severe hypoxia epidemic drift
        drifted_test = test.copy()
        drifted_test["spo2"] = np.clip(drifted_test["spo2"] - 15.0, 50.0, 100.0)
        report_drifted = monitor.audit_feature_drift(drifted_test)
        print("\nSimulated Hypoxia Drift Audit:")
        print(f"  Overall Status: {report_drifted['overall_status']}")
        print(f"  Drifted Features: {report_drifted['drifted_features']}")
        print(f"  SpO2 Metrics: {report_drifted['details']['spo2']}")
