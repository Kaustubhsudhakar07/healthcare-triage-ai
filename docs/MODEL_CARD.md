# Model Card: AI-Assisted Pre-Hospital Patient Criticality Prediction System

## Model Details
- **Model Name:** Pre-Hospital Patient Criticality & Triage Estimator (v1.0.0)
- **Model Architecture:** Tuned XGBoost Regressor with domain-specific clinical preprocessing pipeline (`ClinicalFeatureEngineer` + `ColumnTransformer` + `RobustScaler` + `OneHotEncoder`).
- **Primary Task:** Continuous Acuity Estimation ($1.0 - 10.0$) and 5-Tier Operational Triage Classification (`Low`, `Moderate`, `Elevated`, `High`, `Critical`).
- **Developers:** Healthcare Machine Learning Engineering Portfolio Initiative.
- **Release Date:** August 2026.
- **License:** MIT Open Source (Educational & Decision-Support Prototype).

---

## Intended Use
- **Primary Intended Use:** Pre-hospital emergency triage decision-support for paramedics, emergency medical technicians (EMTs), dispatchers, and emergency department triage nurses to rapidly quantify acute physiological distress and prioritize inbound hospital resources.
- **Primary Intended Users:** Paramedics, triage coordinators, clinical machine learning researchers.
- **Out-of-Scope Uses:**
  - Automated or autonomous medical diagnosis (e.g., diagnosing myocardial infarction, appendicitis, or intracranial hemorrhage).
  - Autonomous patient admission, discharge, or denial of care without clinician review.
  - Medication or surgical protocol prescription.
  - Real-world clinical deployment without prospective clinical trial validation and FDA/CE-MDR regulatory clearance.

---

## Training Data & Population
- **Cohort Size:** 10,000 synthetically modeled pre-hospital patient encounters (8,000 train / 2,000 test).
- **Features (21 predictive inputs):**
  - Demographics & Transit: `age`, `sex`, `ambulance_arrival`.
  - Field Vitals: `heart_rate`, `systolic_bp`, `diastolic_bp`, `spo2`, `respiratory_rate`, `temperature`, `gcs`, `pain_severity`.
  - Physical Signs & Symptoms: `walking_ability`, `altered_consciousness`, `chest_pain`, `difficulty_breathing`, `abdominal_pain`, `injury_trauma`, `bleeding`, `fever`, `headache`, `vomiting`.
  - Context & History: `oxygen_requirement`, `known_cardiac_history`, `known_hypertension`, `known_diabetes`.
- **Target Formulation:** Latent multi-system physiological derangement score ($\Lambda$) with cross-system synergistic non-linearities and Gaussian observation noise, scaled to $[1.0, 10.0]$.

---

## Quantitative Performance Summary (Test Set $N=2,000$)

| Evaluation Metric | Observed Value | Clinical Target Benchmark | Result |
| :--- | :---: | :---: | :---: |
| **$R^2$ Goodness of Fit** | **0.9935** | $> 0.9000$ | Exceeds Target |
| **Mean Absolute Error (MAE)** | **0.1651** score units | $< 0.4000$ | Exceeds Target |
| **Root Mean Squared Error (RMSE)** | **0.2348** score units | $< 0.5500$ | Exceeds Target |
| **Max Error** | **1.50** score units | $< 2.00$ | Passed |
| **Operational Tier Accuracy** | **90.50%** | $> 88.00\%$ | High Fidelity |
| **Macro F1 Score** | **0.8955** | $> 0.8800$ | Balanced |
| **Critical Tier Sensitivity / Recall** | **96.08%** | $> 96.00\%$ | Safety Critical |
| **High Tier Sensitivity / Recall** | **91.19%** | $> 90.00\%$ | Safety Critical |
| **Total Under-Triage Rate** | **4.75%** | $< 5.00\%$ | Minimal Risk |
| **Severe Under-Triage Rate** | **0.00%** | $< 0.50\%$ | Zero Life-Threat Failures |
| **Over-Triage Rate** | **4.75%** | $< 8.00\%$ | Clinically Tolerable |

---

## Subgroup Equity & Robustness Analysis

| Subgroup Slice | Cohort Size (N) | $R^2$ | MAE | Accuracy (%) | Severe Under-Triage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Cohort** | 2,000 | 0.9935 | 0.1651 | 90.50% | **0.0%** |
| **Geriatric (Age $\ge 65$)** | 701 | 0.9938 | 0.1682 | 91.30% | **0.0%** |
| **Young / Adult (Age $< 65$)** | 1,299 | 0.9933 | 0.1634 | 90.07% | **0.0%** |
| **Biological Male** | 1,012 | 0.9941 | 0.1596 | 91.60% | **0.0%** |
| **Biological Female** | 948 | 0.9929 | 0.1696 | 89.35% | **0.0%** |
| **Patients with Cardiac History** | 488 | 0.9924 | 0.1787 | 90.98% | **0.0%** |
| **Hypoxic Cohort ($SpO_2 < 90\%$)** | 531 | 0.9790 | 0.1729 | 93.60% | **0.0%** |
| **Hypotensive Shock ($SBP < 90$)** | 202 | 0.9772 | 0.1777 | 92.57% | **0.0%** |
| **Profound Coma ($GCS \le 8$)** | 162 | 0.9703 | 0.1019 | 95.06% | **0.0%** |

---

## Explainability & Safety Guardrails
1. **Local & Global SHAP Integration:** `shap.TreeExplainer` generates individual patient factor breakdowns, explaining why a specific patient received their score.
2. **Hard Clinical Safety Guardrails:** Immediate override flags trigger for:
   - Coma / Airway Risk: $GCS \le 8$
   - Severe Hypoxemia: $SpO_2 < 88\%$
   - Decompensated Shock: $SBP < 85$ mmHg or Shock Index $\ge 1.2$
   - Hemorrhagic Trauma: Active bleeding with shock
   - Acute Coronary Suspicion: Chest pain in elderly patient with known cardiac history.

---

## Limitations & Ethical Considerations
1. **Synthetic Data Foundation:** The model is trained on a simulated dataset modeling non-linear physiological interactions. While physically coherent, real EHR distributions contain additional unmeasured nuances.
2. **Human-in-the-Loop Mandate:** This software is an advisory support tool. Paramedic and physician judgment must never be subordinated to algorithmic outputs.
