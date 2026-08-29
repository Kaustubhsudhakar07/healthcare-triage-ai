"""
Model Training, Multi-Model Benchmarking, and Hyperparameter Tuning Suite
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Trains and benchmarks multiple model architectures on clinical pre-hospital data.
Evaluates both continuous acuity metrics (MAE, RMSE, R2) and clinical triage safety
metrics (Under-Triage Rate, Severe Under-Triage Rate, Critical Recall, Macro F1).
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath("."))

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    max_error,
    accuracy_score,
    f1_score,
    recall_score,
    classification_report
)

from src.preprocessing import (
    ClinicalFeatureEngineer,
    build_preprocessor,
    build_full_pipeline,
    score_to_urgency_tier,
    clip_criticality_scores,
    TARGET_CONTINUOUS,
    TARGET_CATEGORICAL,
    ID_COLUMN
)


URGENCY_ORDER = ["Low", "Moderate", "Elevated", "High", "Critical"]
TIER_MAP = {tier: idx for idx, tier in enumerate(URGENCY_ORDER)}


def compute_triage_safety_metrics(
    y_true_score: np.ndarray,
    y_pred_score: np.ndarray,
    y_true_cat: np.ndarray,
    y_pred_cat: np.ndarray
) -> Dict[str, float]:
    """
    Computes both continuous and clinical safety classification metrics.
    """
    # Continuous metrics
    mae = mean_absolute_error(y_true_score, y_pred_score)
    rmse = np.sqrt(mean_squared_error(y_true_score, y_pred_score))
    r2 = r2_score(y_true_score, y_pred_score)
    max_err = max_error(y_true_score, y_pred_score)
    
    # Classification metrics
    acc = accuracy_score(y_true_cat, y_pred_cat)
    macro_f1 = f1_score(y_true_cat, y_pred_cat, labels=URGENCY_ORDER, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_cat, y_pred_cat, labels=URGENCY_ORDER, average="weighted", zero_division=0)
    
    # Per-class recall
    recalls = recall_score(y_true_cat, y_pred_cat, labels=URGENCY_ORDER, average=None, zero_division=0)
    crit_recall = recalls[TIER_MAP["Critical"]]
    high_recall = recalls[TIER_MAP["High"]]
    
    # Clinical Under-Triage and Over-Triage calculations
    true_ranks = np.array([TIER_MAP[c] for c in y_true_cat])
    pred_ranks = np.array([TIER_MAP[c] for c in y_pred_cat])
    
    under_triage = np.mean(pred_ranks < true_ranks) * 100.0
    over_triage = np.mean(pred_ranks > true_ranks) * 100.0
    
    # Severe Under-Triage: Actual is High/Critical (>=3), but Predicted is Low/Moderate (<=1)
    is_high_acuity = true_ranks >= TIER_MAP["High"]
    is_pred_low_acuity = pred_ranks <= TIER_MAP["Moderate"]
    if np.sum(is_high_acuity) > 0:
        severe_under_triage = (np.sum(is_high_acuity & is_pred_low_acuity) / np.sum(is_high_acuity)) * 100.0
    else:
        severe_under_triage = 0.0

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "Max_Error": round(float(max_err), 4),
        "Accuracy_%": round(float(acc * 100.0), 2),
        "Macro_F1": round(float(macro_f1), 4),
        "Weighted_F1": round(float(weighted_f1), 4),
        "Critical_Recall_%": round(float(crit_recall * 100.0), 2),
        "High_Recall_%": round(float(high_recall * 100.0), 2),
        "Under_Triage_%": round(float(under_triage), 2),
        "Severe_Under_Triage_%": round(float(severe_under_triage), 2),
        "Over_Triage_%": round(float(over_triage), 2)
    }


def get_model_candidates(seed: int = 42) -> Dict[str, Any]:
    """
    Returns candidate regression estimators for benchmarking.
    """
    return {
        "Ridge_Regression": Ridge(alpha=1.0, random_state=seed),
        "Decision_Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=seed),
        "Random_Forest": RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=seed),
        "Extra_Trees": ExtraTreesRegressor(n_estimators=150, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=seed),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.06, max_depth=5, subsample=0.85, colsample_bytree=0.85, random_state=seed, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=150, learning_rate=0.06, max_depth=6, num_leaves=31, subsample=0.85, random_state=seed, n_jobs=-1, verbose=-1)
    }


def run_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains and benchmarks all candidate models.
    """
    feature_cols = [c for c in train_df.columns if c not in [ID_COLUMN, TARGET_CONTINUOUS, TARGET_CATEGORICAL]]
    
    X_train = train_df[feature_cols]
    y_train_score = train_df[TARGET_CONTINUOUS].values
    
    X_test = test_df[feature_cols]
    y_test_score = test_df[TARGET_CONTINUOUS].values
    y_test_cat = test_df[TARGET_CATEGORICAL].values
    
    # 1. Transform features with preprocessor
    fe = ClinicalFeatureEngineer()
    X_train_eng = fe.fit_transform(X_train)
    X_test_eng = fe.transform(X_test)
    
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_eng)
    X_test_proc = preprocessor.transform(X_test_eng)
    
    models = get_model_candidates(seed=seed)
    
    # Add Stacking/Voting Ensemble of top tree & gradient boosted models
    voting_ensemble = VotingRegressor(
        estimators=[
            ("xgb", XGBRegressor(n_estimators=150, learning_rate=0.06, max_depth=5, random_state=seed, n_jobs=-1)),
            ("lgb", LGBMRegressor(n_estimators=150, learning_rate=0.06, max_depth=6, random_state=seed, n_jobs=-1, verbose=-1)),
            ("rf", RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=4, random_state=seed, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=seed))
        ],
        n_jobs=-1
    )
    models["Voting_Ensemble"] = voting_ensemble
    
    results = []
    fitted_models = {}
    
    print("\n" + "="*80)
    print("RUNNING MULTI-MODEL BENCHMARK EVALUATION (5-Fold CV + Test Partition)")
    print("="*80)
    
    for name, estimator in models.items():
        print(f"\nTraining model: {name}...")
        
        # 5-Fold Cross Validation on Train Set
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_r2_scores = cross_val_score(estimator, X_train_proc, y_train_score, cv=kf, scoring="r2", n_jobs=-1)
        cv_mae_scores = -cross_val_score(estimator, X_train_proc, y_train_score, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
        
        # Fit on full training set
        estimator.fit(X_train_proc, y_train_score)
        fitted_models[name] = estimator
        
        # Evaluate on Test Set
        y_pred_raw = estimator.predict(X_test_proc)
        y_pred_score = clip_criticality_scores(y_pred_raw)
        y_pred_cat = score_to_urgency_tier(y_pred_score)
        
        metrics = compute_triage_safety_metrics(y_test_score, y_pred_score, y_test_cat, y_pred_cat)
        metrics["Model"] = name
        metrics["CV_R2_Mean"] = round(float(np.mean(cv_r2_scores)), 4)
        metrics["CV_R2_Std"] = round(float(np.std(cv_r2_scores)), 4)
        metrics["CV_MAE_Mean"] = round(float(np.mean(cv_mae_scores)), 4)
        
        results.append(metrics)
        print(f"  -> Test R2: {metrics['R2']:.4f} | Test MAE: {metrics['MAE']:.4f} | Accuracy: {metrics['Accuracy_%']}% | Severe Under-Triage: {metrics['Severe_Under_Triage_%']}%")

    results_df = pd.DataFrame(results)
    
    # Order columns
    primary_cols = [
        "Model", "R2", "MAE", "RMSE", "Max_Error",
        "CV_R2_Mean", "CV_MAE_Mean", "Accuracy_%", "Macro_F1",
        "Critical_Recall_%", "High_Recall_%",
        "Under_Triage_%", "Severe_Under_Triage_%", "Over_Triage_%"
    ]
    results_df = results_df[primary_cols].sort_values(by="R2", ascending=False).reset_index(drop=True)
    
    return results_df, fitted_models


def tune_best_model(
    train_df: pd.DataFrame,
    seed: int = 42
) -> Any:
    """
    Performs hyperparameter tuning via RandomizedSearchCV for XGBoost.
    """
    print("\n" + "="*80)
    print("RUNNING HYPERPARAMETER OPTIMIZATION (RandomizedSearchCV on XGBoost)")
    print("="*80)
    
    feature_cols = [c for c in train_df.columns if c not in [ID_COLUMN, TARGET_CONTINUOUS, TARGET_CATEGORICAL]]
    X_train = train_df[feature_cols]
    y_train_score = train_df[TARGET_CONTINUOUS].values
    
    fe = ClinicalFeatureEngineer()
    X_train_eng = fe.fit_transform(X_train)
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_eng)
    
    param_dist = {
        "n_estimators": [100, 150, 200, 250],
        "learning_rate": [0.03, 0.05, 0.08, 0.10],
        "max_depth": [4, 5, 6, 7],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.75, 0.85, 0.95],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2]
    }
    
    base_xgb = XGBRegressor(random_state=seed, n_jobs=-1)
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring="r2",
        random_state=seed,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train_proc, y_train_score)
    print(f"Best CV R2: {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")
    
    return search.best_estimator_, search.best_params_


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark clinical triage models.")
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv")
    parser.add_argument("--test_path", type=str, default="data/processed/test.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.experiments_dir, exist_ok=True)
    
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    # 1. Run Benchmark
    benchmark_df, fitted_models = run_benchmark(train_df, test_df, seed=args.seed)
    
    benchmark_csv_path = os.path.join(args.experiments_dir, "benchmark_results.csv")
    benchmark_df.to_csv(benchmark_csv_path, index=False)
    print(f"\nBenchmark results written to: {benchmark_csv_path}")
    print("\n" + benchmark_df.to_string(index=False))
    
    # 2. Hyperparameter Tuning on top model
    best_tuned_model, best_params = tune_best_model(train_df, seed=args.seed)
    
    # 3. Construct Full End-to-End Pipeline with best tuned model
    print("\nConstructing Full End-to-End Pipeline...")
    full_pipeline = build_full_pipeline(best_tuned_model)
    
    feature_cols = [c for c in train_df.columns if c not in [ID_COLUMN, TARGET_CONTINUOUS, TARGET_CATEGORICAL]]
    X_train = train_df[feature_cols]
    y_train_score = train_df[TARGET_CONTINUOUS].values
    
    full_pipeline.fit(X_train, y_train_score)
    
    # Final Test Set Evaluation
    X_test = test_df[feature_cols]
    y_test_score = test_df[TARGET_CONTINUOUS].values
    y_test_cat = test_df[TARGET_CATEGORICAL].values
    
    y_pred_raw = full_pipeline.predict(X_test)
    y_pred_score = clip_criticality_scores(y_pred_raw)
    y_pred_cat = score_to_urgency_tier(y_pred_score)
    
    final_metrics = compute_triage_safety_metrics(y_test_score, y_pred_score, y_test_cat, y_pred_cat)
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION FOR BEST PRODUCTION PIPELINE (Tuned XGBoost):")
    print("="*80)
    for k, v in final_metrics.items():
        print(f"  - {k}: {v}")
        
    # Save Pipeline and Model Artifacts
    pipeline_path = os.path.join(args.models_dir, "pipeline.joblib")
    best_model_path = os.path.join(args.models_dir, "best_model.joblib")
    metadata_path = os.path.join(args.models_dir, "model_metadata.json")
    
    joblib.dump(full_pipeline, pipeline_path)
    joblib.dump(best_tuned_model, best_model_path)
    
    metadata = {
        "model_type": "Tuned XGBoost Regressor with Clinical Pipeline",
        "best_hyperparameters": best_params,
        "metrics_test": final_metrics,
        "urgency_categories": URGENCY_ORDER,
        "feature_count": len(feature_cols),
        "target_variable": "criticality_score (continuous 1.0 - 10.0)",
        "mapping": "Urgency Category mapped via physiological acuity intervals"
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nSerialized artifacts:")
    print(f"  [+] Full Pipeline:   {pipeline_path}")
    print(f"  [+] Best Model:      {best_model_path}")
    print(f"  [+] Metadata JSON:   {metadata_path}")


if __name__ == "__main__":
    main()
