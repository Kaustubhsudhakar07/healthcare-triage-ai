"""
Data Schema and Physiological Validation Suite
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Validates schema types, null rates, physiological range boundaries, and target invariants.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


EXPECTED_COLUMNS = [
    "patient_id",
    "age",
    "sex",
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
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature",
    "gcs",
    "pain_severity",
    "oxygen_requirement",
    "known_cardiac_history",
    "known_hypertension",
    "known_diabetes",
    "criticality_score",
    "urgency_category"
]

BINARY_COLUMNS = [
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

PHYSIOLOGICAL_BOUNDS = {
    "age": (16, 95),
    "heart_rate": (30.0, 250.0),
    "systolic_bp": (50.0, 260.0),
    "diastolic_bp": (30.0, 160.0),
    "spo2": (50.0, 100.0),
    "respiratory_rate": (6.0, 60.0),
    "temperature": (33.0, 43.0),
    "gcs": (3, 15),
    "pain_severity": (0, 10),
    "criticality_score": (1.0, 10.0)
}

VALID_SEX_VALUES = {"Male", "Female", "Other"}
VALID_URGENCY_VALUES = {"Low", "Moderate", "Elevated", "High", "Critical"}


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates a dataset against schema and physiological consistency rules.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with validation results, errors, and warnings.
        
    Raises
    ------
    ValueError
        If any critical schema violation or physiological impossibility is detected.
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # 1. Column presence check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing mandatory columns: {sorted(list(missing_cols))}")
    
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        warnings.append(f"Extra columns detected: {sorted(list(extra_cols))}")
        
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
        
    # 2. Check for missing values in critical columns
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        warnings.append(f"Columns with null values detected: {columns_with_nulls.to_dict()}")
        
    # 3. Check binary columns
    for col in BINARY_COLUMNS:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            errors.append(f"Binary column '{col}' has invalid values: {unique_vals}")
            
    # 4. Check categorical values
    sex_vals = set(df["sex"].dropna().unique())
    if not sex_vals.issubset(VALID_SEX_VALUES):
        errors.append(f"Column 'sex' contains invalid categories: {sex_vals - VALID_SEX_VALUES}")
        
    if "urgency_category" in df.columns:
        urgency_vals = set(df["urgency_category"].dropna().unique())
        if not urgency_vals.issubset(VALID_URGENCY_VALUES):
            errors.append(f"Column 'urgency_category' contains invalid categories: {urgency_vals - VALID_URGENCY_VALUES}")

    # 5. Check physiological numerical boundaries
    for col, (lower, upper) in PHYSIOLOGICAL_BOUNDS.items():
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < lower or max_val > upper:
                errors.append(
                    f"Physiological out-of-bounds in '{col}': observed [{min_val:.1f}, {max_val:.1f}], "
                    f"allowed [{lower}, {upper}]"
                )
                
    # 6. Physiological invariant check: SBP must be strictly greater than DBP
    if "systolic_bp" in df.columns and "diastolic_bp" in df.columns:
        invalid_bp = df[df["systolic_bp"] <= df["diastolic_bp"]]
        if len(invalid_bp) > 0:
            errors.append(f"Physiological anomaly: {len(invalid_bp)} records have SBP <= DBP.")
            
    # 7. Patient ID uniqueness
    if "patient_id" in df.columns:
        if df["patient_id"].duplicated().any():
            errors.append("Duplicate patient_id values detected.")

    # 8. Evaluation of target score vs category consistency
    if "criticality_score" in df.columns and "urgency_category" in df.columns:
        mismatches = 0
        for _, row in df.iterrows():
            score = row["criticality_score"]
            cat = row["urgency_category"]
            expected_cat = (
                "Low" if score < 2.5 else
                "Moderate" if score < 4.5 else
                "Elevated" if score < 6.5 else
                "High" if score < 8.5 else
                "Critical"
            )
            if cat != expected_cat:
                mismatches += 1
        if mismatches > 0:
            errors.append(f"Target inconsistency: {mismatches} records have mismatched criticality_score and urgency_category.")

    result = {
        "status": "PASSED" if not errors else "FAILED",
        "total_records": len(df),
        "total_columns": len(df.columns),
        "errors": errors,
        "warnings": warnings
    }
    
    if errors:
        raise ValueError(f"Dataset validation failed with {len(errors)} error(s):\n" + "\n".join(f" - {e}" for e in errors))
        
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate patient criticality dataset.")
    parser.add_argument("--data_path", type=str, default="data/raw/patient_criticality_data.csv", help="Path to CSV dataset.")
    args = parser.parse_args()
    
    print(f"Loading and validating dataset from '{args.data_path}'...")
    try:
        df = pd.read_csv(args.data_path)
        result = validate_dataset(df)
        print("\n=======================================================")
        print("[PASS] DATASET VALIDATION PASSED SUCCESSFULLY!")
        print("=======================================================")
        print(f"Total Records: {result['total_records']:,}")
        print(f"Total Columns: {result['total_columns']}")
        if result["warnings"]:
            print(f"Warnings ({len(result['warnings'])}):")
            for w in result["warnings"]:
                print(f"  [WARN] {w}")
        else:
            print("No warnings or physiological anomalies detected.")
    except Exception as e:
        print("\n=======================================================")
        print("[FAIL] DATASET VALIDATION FAILED!")
        print("=======================================================")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
