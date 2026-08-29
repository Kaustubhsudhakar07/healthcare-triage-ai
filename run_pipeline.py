"""
Master Pipeline Orchestrator
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Executes the complete machine learning lifecycle from scratch:
1. Synthetic Data Generation (N=10,000)
2. Data Schema & Physiological Boundary Validation
3. Stratified Train/Test Dataset Splitting
4. Multi-Model Training, Benchmarking & Hyperparameter Tuning
5. Clinical Error Analysis, Subgroup Audits & Visualization Plots
6. SHAP Explainability Artifact Generation
7. Automated Test Suite Execution (pytest)
"""

import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.abspath("."))


def run_stage(step_num: int, stage_name: str, command: list):
    """Executes a pipeline stage with timing and status output."""
    print("\n" + "=" * 80)
    print(f"[STAGE {step_num}/7] {stage_name.upper()}")
    print("=" * 80)
    
    t0 = time.time()
    result = subprocess.run([sys.executable] + command, capture_output=False)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n❌ [STAGE {step_num} FAILED] {stage_name} (Code: {result.returncode})")
        sys.exit(result.returncode)
    else:
        print(f"✅ [STAGE {step_num} COMPLETE] {stage_name} ({elapsed:.2f}s)")


def main():
    print("""
    ================================================================================
    🚑 PRE-HOSPITAL PATIENT CRITICALITY & EMERGENCY TRIAGE SUPPORT SYSTEM
    ================================================================================
    """)
    total_start = time.time()
    
    # Stage 1: Generate Data
    run_stage(1, "Synthetic Patient Data Generation (N=10,000)", ["src/generate_data.py", "--n_samples", "10000", "--seed", "42"])
    
    # Stage 2: Validate Data
    run_stage(2, "Data Schema & Physiological Range Validation", ["src/data_validation.py", "--data_path", "data/raw/patient_criticality_data.csv"])
    
    # Stage 3: Split Data
    run_stage(3, "Stratified Train/Test Dataset Partitioning", ["src/data_split.py", "--test_size", "0.20", "--seed", "42"])
    
    # Stage 4: Multi-Model Benchmark & Hyperparameter Tuning
    run_stage(4, "Multi-Model Training & XGBoost Tuning", ["src/train.py", "--seed", "42"])
    
    # Stage 5: Evaluation & Subgroup Safety Audit
    run_stage(5, "Clinical Safety Auditing & Figure Generation", ["src/evaluate.py"])
    
    # Stage 6: SHAP Explainability
    run_stage(6, "Global SHAP Explainability Visualizations", ["src/explainability.py"])
    
    # Stage 7: Automated Test Suite
    run_stage(7, "Unit and Integration Testing (pytest)", ["-m", "pytest", "tests/", "-v"])
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print("=" * 80)
    print("\nTo start the interactive Streamlit Command Center, run:")
    print("   streamlit run app.py\n")
    print("To start the FastAPI production microservice, run:")
    print("   uvicorn src.api:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
