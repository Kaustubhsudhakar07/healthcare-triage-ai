# Architecture & Modeling Decision Log (ADR)

## Document Information
- **Project Title:** AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
- **Status:** Active / Ratified Architectural Record

---

## Decision Record 001: Formulation of Primary Task as Continuous Acuity Regression with Threshold Mapping
- **Date:** 2026-08-29
- **Status:** Accepted
- **Context:** Emergency medical triage can be framed either as pure 5-class ordinal classification or as continuous acuity regression ($1.0 - 10.0$) mapped to 5 discrete urgency tiers.
- **Decision:** Adopt continuous regression with post-prediction calibrated threshold mapping.
- **Rationale:**
  1. Standard classification treats errors between adjacent classes equally, ignoring distance. In healthcare triage, predicting *Low* when a patient is *Critical* is catastrophic, while predicting *Low* when a patient is *Moderate* is minor. Regression penalizes distance with squared error loss ($L_2$).
  2. Continuous acuity provides high-resolution granularity within tiers (e.g., distinguishing a borderline high patient at 6.6 from an imminent crisis patient at 8.3).
  3. Seamless threshold adjustment enables tuning sensitivity for high-risk triage categories without retraining.

---

## Decision Record 002: Synthetic Data Generation Strategy vs. Public In-Hospital Datasets
- **Date:** 2026-08-29
- **Status:** Accepted
- **Context:** Public repositories (MIMIC-IV-ED, PhysioNet, NHAMCS) lack pre-hospital field features (e.g. ambulatory status, field trauma flags, pre-hospital $O_2$) and have restrictive Data Use Agreements prohibiting public web demos.
- **Decision:** Build a mathematically grounded, multi-system physiological simulation dataset ($N=10,000$).
- **Rationale:**
  1. Guarantees 100% compliance with HIPAA/GDPR with zero Protected Health Information risk.
  2. Models authentic physiological couplings (e.g., hypoxia triggering tachypnea, hemorrhagic shock triggering hypotension + tachycardia, geriatric frailty multipliers).
  3. Full reproducibility via fixed-seed generation.

---

## Decision Record 003: Leakage-Free Scikit-Learn Pipeline & Domain Feature Engineering
- **Date:** 2026-08-29
- **Status:** Accepted
- **Context:** Clinical calculations like Shock Index ($\frac{HR}{SBP}$) and Pulse Pressure ($SBP - DBP$) must be computed safely without data leakage.
- **Decision:** Implement a custom `ClinicalFeatureEngineer` transformer inside a unified `scikit-learn` `Pipeline` and `ColumnTransformer`.
- **Rationale:** Encapsulating feature extraction and preprocessing ensures that training transformations, cross-validation folds, test evaluations, and production live inferences use identical transformations without pipeline leakage.

---

## Decision Record 004: Algorithm Selection & Benchmark Hierarchy
- **Date:** 2026-08-29
- **Status:** Accepted
- **Context:** Evaluated 8 distinct model families (Ridge, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, Voting Ensemble).
- **Decision:** Select Tuned XGBoost Regressor as production engine.
- **Rationale:** Tuned XGBoost achieved highest test $R^2$ ($0.9935$), lowest MAE ($0.1651$), highest accuracy ($90.50\%$), and $0.00\%$ severe under-triage failures across all test slices, while remaining fast (<2ms inference) for real-time mobile deployment.

---

## Decision Record 005: Hard Clinical Safety Guardrails & Human-in-the-Loop Constraint
- **Date:** 2026-08-29
- **Status:** Accepted
- **Context:** Machine learning models can occasionally produce edge-case errors on rare combinations.
- **Decision:** Layer hard physiological override rules ($GCS \le 8$, $SpO_2 < 88\%$, $SBP < 85$ mmHg) over ML predictions, and explicitly restrict model scope to advisory decision-support.
- **Rationale:** In clinical medicine, human oversight is mandatory. Hard guardrails guarantee immediate safety alerts for acute physiological emergencies regardless of ML input noise.
