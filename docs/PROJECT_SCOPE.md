# Project Scope: Boundaries, Responsibilities & Operational Limits

## Document Information
- **Project Title:** AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
- **Document Version:** 1.0.0
- **Document Status:** Active / Ratified Scope Definition

---

## 1. Project Purpose

The purpose of this project is to build an end-to-end, production-grade machine learning pipeline and interactive web interface that demonstrates how pre-hospital clinical parameters can be used to estimate patient criticality and prioritize emergency care.

---

## 2. In-Scope Components & Deliverables

| Area | In-Scope Deliverables & Features |
| :--- | :--- |
| **Data Generation & Quality** | - Scientifically modeled synthetic dataset generation ($N = 10,000$ patient records).<br>- Incorporation of realistic non-linear physiological interactions, physiological noise, missingness, and measurement variances.<br>- Comprehensive data validation script (`src/data_validation.py`) checking schema, ranges, anomalies, and physiological boundaries. |
| **Exploratory Data Analysis** | - Statistical analysis of vital distributions, symptom correlations, target distributions, and missing data patterns in `notebooks/01_eda.ipynb`. |
| **Preprocessing & Pipelines** | - Leakage-free `scikit-learn` `Pipeline` and `ColumnTransformer` architecture fitted strictly on training splits.<br>- Robust imputation (median for skewed vitals, most frequent/constant for categorical indicators).<br>- Standardized numerical scaling and categorical encoding. |
| **Modeling & Benchmarking** | - Multi-model training and rigorous benchmarking across: Baseline (Logistic / Ridge Regression), Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, and Stacking/Voting Ensemble.<br>- Quantitative evaluation across multiple metrics (Regression: MAE, RMSE, $R^2$; Classification: Macro F1, Weighted F1, Recall for High/Critical classes).<br>- Systematic Hyperparameter Tuning via Cross-Validation (`GridSearchCV` / `RandomizedSearchCV`). |
| **Evaluation & Error Analysis** | - Exhaustive error analysis studying false negatives (under-triage risk) and false positives (over-triage risk) in `docs/ERROR_ANALYSIS.md`. |
| **Explainable AI (XAI)** | - Global and local feature attributions using SHAP (SHapley Additive exPlanations) or TreeExplainer to explain individual patient criticality predictions. |
| **Inference Engine** | - Clean, modular inference service (`src/predict.py`) handling payload validation, pipeline transformation, model execution, threshold mapping, and explanation generation. |
| **Frontend Web App** | - Professional, intuitive, dark-mode Streamlit user interface (`app.py`) for real-time field triage data entry, confidence display, and visual explanation charts. |
| **Testing & Quality Assurance** | - Comprehensive test suite with `pytest` covering data validation, preprocessing transformations, prediction outputs, score range invariants ($[1.0, 10.0]$), and error handling. |
| **DevOps & Deployment** | - Containerization with `Dockerfile` and `.dockerignore`.<br>- Cloud deployment guidance for Streamlit Community Cloud and Container platforms (e.g., Render, Railway, AWS). |
| **Comprehensive Documentation** | - Exhaustive architectural guides, decision logs (`docs/DECISIONS.md`), model card (`docs/MODEL_CARD.md`), clinical assumptions, security/privacy documentation, and job interview Q&A. |

---

## 3. Out-of-Scope Components & Hard Exclusions

| Domain | Excluded Activity / Feature | Justification |
| :--- | :--- | :--- |
| **Disease Diagnosis** | Diagnosing specific medical conditions (e.g., "Patient has Appendicitis"). | Pre-hospital vital signs and broad symptoms lack specificity for definitive disease diagnosis. |
| **Treatment & Medication** | Suggesting drug regimens, dosages, or surgical protocols. | Prescribing medical therapy requires licensed clinical judgement, detailed allergy history, and diagnostic confirmation. |
| **Autonomous Clinical Decision-Making** | Automatically admitting, discharging, or refusing care to patients without paramedic/physician review. | Safety-critical healthcare algorithms must operate strictly under human-in-the-loop oversight. |
| **Emergency 911/999 CAD Dispatch Automation** | Directly dispatching emergency vehicles without human emergency call-taker authorization. | Dispatch protocols involve geographic, resource, and telecommunication constraints outside tabular ML scope. |
| **Real Patient PHI Processing** | Ingesting un-anonymized real patient Protected Health Information (PHI) in this prototype. | Strict compliance with HIPAA/GDPR regulations prevents using raw identifiable hospital data in an open portfolio codebase. |
| **Claims of Clinical Validation** | Advertising the $1–10$ score as a medically validated scoring system (like APACHE-II or SOFA). | The synthetic acuity index is an engineering target created for ML demonstration, not an internationally validated medical index. |

---

## 4. Scope Governance & Decision Protocol

Any architectural, mathematical, or structural change to the project must be evaluated against this scope document. If a proposed feature encroaches upon the Out-of-Scope list (e.g., attempting to predict disease codes), it must be rejected immediately and documented in `docs/DECISIONS.md`.
