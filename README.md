# AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.0-red.svg)](https://xgboost.readthedocs.io/)
[![Google Gemini](https://img.shields.io/badge/GenAI-Google%20Gemini-4285f4.svg)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/RAG-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Testing-33%20Passed-brightgreen.svg)](https://pytest.org/)

An end-to-end Machine Learning Engineering & Agentic AI decision-support platform that predicts a standardized **Patient Criticality Score ($1.0 - 10.0$)**, maps it into a 5-tier **Operational Emergency Urgency Category (Low, Moderate, Elevated, High, Critical)**, and integrates a conversational **Triage AI Agent** powered by Google Gemini, real ML what-if sensitivity analysis, and ChromaDB clinical literature grounding.

---

## 🚑 Key Capabilities & Architecture

1. **Scientifically Modeled Acuity Formulation ($N=10,000$):**
   - Built on physiological modeling of respiratory derangement ($SpO_2$, RR, dyspnea), hemodynamic shock (Shock Index, hypotension, hemorrhage), neurological coma ($GCS \le 8$, walking status), and systemic inflammation.
2. **Leakage-Free Clinical Pipeline:**
   - Encapsulated `scikit-learn` `Pipeline` with automated feature engineering (`Shock Index`, `Pulse Pressure`, `Hypoxia Flag`, `Coma Flag`, `Geriatric Risk`), robust scaling, and categorical one-hot encoding.
3. **Rigorous Multi-Model Benchmarking:**
   - Benchmarked across 8 model families with 5-Fold Cross Validation: Tuned XGBoost ($R^2 = 0.9935$, $\text{MAE} = 0.1651$, accuracy $90.50\%$, severe under-triage $0.00\%$).
4. **Agentic Triage AI Assistant (Google Gemini):**
   - Single primary agent with 7 controlled tools answering clinical questions, explaining SHAP attribution factors, running real ML what-if sensitivity analyses, and explaining system architecture.
5. **Real ML What-If Sensitivity Engine:**
   - Executes true sensitivity analysis: modifies only the requested parameter in the current patient payload and re-runs the **real** trained XGBoost pipeline, reporting exact mathematical deltas. The LLM never hallucinates numbers.
6. **ChromaDB Clinical Literature RAG:**
   - Vector store indexing authoritative guidelines from NIH (StatPearls), WHO, CDC, and NHS (NEWS2) with verified citations.
7. **Clinical Safety Guardrails:**
   - Pre-screen regex filters refusing diagnosis, medication prescriptions, and advice to bypass emergency care, with adversarial prompt injection resistance.
8. **Interactive Command Center Web App (6 Tabs):**
   - Live Triage Console, Triage AI Assistant, Inbound Ambulance Queue, Model Benchmarks, Real-Time Data Drift Monitor, and Clinical Scope.

---

## 📊 Benchmark Performance Summary (Test Set $N=2,000$)

| Model Architecture | $R^2$ Score | MAE | RMSE | Accuracy (%) | Macro F1 | Critical Recall (%) | Severe Under-Triage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost (Production)** | **0.9935** | **0.1651** | **0.2348** | **90.50%** | **0.8955** | **96.08%** | **0.00%** |
| LightGBM Regressor | 0.9923 | 0.1751 | 0.2553 | 90.05% | 0.8889 | 96.86% | 0.00% |
| Gradient Boosting Regressor | 0.9922 | 0.1770 | 0.2566 | 90.05% | 0.8899 | 95.88% | 0.00% |
| Voting Ensemble | 0.9909 | 0.1893 | 0.2773 | 89.40% | 0.8813 | 96.47% | 0.00% |
| Extra Trees Regressor | 0.9851 | 0.2412 | 0.3549 | 88.00% | 0.8643 | 93.92% | 0.00% |
| Random Forest Regressor | 0.9821 | 0.2627 | 0.3891 | 86.60% | 0.8465 | 94.31% | 0.00% |
| Decision Tree Regressor | 0.9637 | 0.3762 | 0.5544 | 79.40% | 0.7683 | 93.33% | 0.24% |
| Ridge Regression (Baseline) | 0.9604 | 0.4204 | 0.5785 | 77.90% | 0.7652 | 76.86% | 0.00% |

---

## 📁 Repository Structure

```
HealthCare_Hospital_patient_critical_score/
├── data/
│   ├── raw/
│   │   └── patient_criticality_data.csv       # N=10,000 synthetic patient dataset
│   └── processed/
│       ├── train.csv                          # 80% Stratified Training Split (N=8,000)
│       └── test.csv                           # 20% Stratified Test Split (N=2,000)
├── docs/
│   ├── PROJECT_SCOPE.md                       # Scope boundaries & operational limits
│   ├── PROJECT_REQUIREMENTS.md                # Problem formulation & clinical inputs
│   ├── TARGET_DEFINITION.md                   # Latent physiological acuity mathematics
│   ├── DATASET_INVESTIGATION.md               # Empirical public dataset feasibility audit
│   ├── FEATURE_DICTIONARY.md                  # 21 predictive features schema
│   ├── ERROR_ANALYSIS.md                      # Triage safety audit & failure mode diagnostics
│   ├── MODEL_CARD.md                          # Standardized AI Model Card
│   └── DECISIONS.md                           # Architectural Decision Records (ADR)
├── experiments/
│   └── benchmark_results.csv                  # Multi-model benchmarking metrics log
├── models/
│   ├── pipeline.joblib                        # Production end-to-end inference pipeline
│   ├── best_model.joblib                      # Tuned XGBoost estimator
│   └── model_metadata.json                    # Serialized metadata & parameters
├── reports/
│   └── figures/                               # Diagnostic evaluation figures & SHAP plots
│       ├── predicted_vs_actual.png
│       ├── residual_analysis.png
│       ├── confusion_matrix.png
│       ├── under_triage_by_acuity.png
│       ├── shap_summary_beeswarm.png
│       └── shap_feature_importance_bar.png
├── src/
│   ├── __init__.py
│   ├── generate_data.py                       # Synthetic data generator
│   ├── data_validation.py                     # Physiological validation suite
│   ├── preprocessing.py                       # Leakage-free ColumnTransformer pipeline
│   ├── data_split.py                          # Stratified train/test splitting
│   ├── train.py                               # Training, CV benchmarking & hyperparameter tuning
│   ├── evaluate.py                            # Evaluation metrics & error analysis generator
│   ├── explainability.py                      # SHAP global & local attribution engine
│   └── predict.py                             # Production inference engine with safety guardrails
├── tests/
│   ├── __init__.py
│   ├── test_data.py                           # Unit tests for data generation & schema
│   ├── test_pipeline.py                       # Unit tests for preprocessor & transformations
│   └── test_inference.py                      # Unit tests for inference & guardrails
├── app.py                                     # Interactive dark-mode Streamlit web app
├── requirements.txt                           # Python dependencies
├── Dockerfile                                 # Container build configuration
├── .dockerignore
└── README.md
```

---

## 🚀 Quick Start & Execution

### 1. Environment Setup
```bash
# Clone repository and create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Generation & Validation
```bash
python src/generate_data.py
python src/data_validation.py
```

### 3. Run Pipeline Training & Benchmarking
```bash
python src/data_split.py
python src/train.py
python src/evaluate.py
python src/explainability.py
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## ⚠️ Mandatory Research & Portfolio Disclaimer
This system is an **educational machine learning engineering decision-support prototype**. It is trained on a synthetic dataset ($N = 10,000$) designed to simulate pre-hospital physiological triage dynamics. It has **not** received regulatory clearance (e.g., FDA 510(k), CE mark, EU MDR) and must **never** be used as an autonomous medical diagnostic device or to replace licensed clinical judgment.
