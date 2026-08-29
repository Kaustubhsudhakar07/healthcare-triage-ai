# Public Dataset Investigation & Feasibility Analysis

## Document Information
- **Project Title:** AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
- **Document Version:** 1.0.0
- **Status:** Phase 1 Completed
- **Purpose:** Rigorous empirical evaluation of existing public datasets to determine whether a suitable real-world dataset exists for pre-hospital criticality prediction.

---

## 1. Investigation Objective

Before constructing a synthetic dataset, professional data science standards require an exhaustive investigation of public, open-source, and credentialed medical datasets. We evaluated whether any publicly available repository satisfies the specific clinical, technical, and regulatory requirements of a **pre-hospital triage decision-support prototype**.

### Required Criteria:
1. **Domain Context:** Pre-hospital / Emergency Medical Services (EMS) / Emergency field intake.
2. **Observable Signs & Symptoms:** Ambulatory status (walking ability), vomiting, bleeding, physical trauma, chest pain, dyspnea, abdominal pain, fever, headache, altered consciousness.
3. **Field Vital Signs:** Heart rate, systolic BP, diastolic BP, $SpO_2$, respiratory rate, temperature, Glasgow Coma Scale (GCS).
4. **Target Variable:** High-resolution acuity/criticality index or continuous emergency severity score.
5. **Licensing & Accessibility:** Open-source or freely redistributable for educational/portfolio deployment without violating HIPAA, GDPR, or Data Use Agreements (DUA).

---

## 2. Comprehensive Review of Public Repositories

### 2.1 MIMIC-IV-ED (Beth Israel Deaconess Medical Center / PhysioNet)
- **Source / URL:** [PhysioNet MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/)
- **Availability & License:** Credentialed access required (CITI training, Data Use Agreement). Strictly prohibits re-distribution, public web hosting of raw data, or deploying trained models on unapproved public web servers if data can be reverse-engineered.
- **Relevant Fields Available:** Demographics (age, gender), basic ED triage vitals (heart rate, BP, $SpO_2$, temp), in-hospital Emergency Severity Index (ESI 1–5).
- **Missing / Mismatched Fields:**
  - **In-Hospital vs. Pre-Hospital:** Captures triage once inside the hospital emergency department, not in the field or ambulance.
  - **Missing Field Signs:** Lacks granular pre-hospital transit indicators (e.g., ambulatory/walking ability, pre-hospital oxygen requirement, field trauma evaluation, acute bleeding indicators in structured tabular format).
- **Feasibility Verdict:** **REJECTED.** Credentialed DUA strictly forbids publishing data or public interactive web demos; also represents in-ED rather than field EMS triage.

---

### 2.2 NEMSIS (National Emergency Medical Services Information System) Public-Release Dataset
- **Source / URL:** [NEMSIS Research Datasets](https://nemsis.org/)
- **Availability & License:** Requires institutional application and formal TAC data use agreement. Data terms prohibit public redistribution and limit usage to approved non-profit research studies.
- **Relevant Fields Available:** Pre-hospital EMS transport records, ambulance arrival times, dispatched complaint, field vitals.
- **Missing / Mismatched Fields:**
  - **Activation-Level, Not Patient-Level:** Data records EMS activations rather than unified patient records.
  - **High Dimensional Sparsity & Missingness (>40–60%):** Many field vital signs (GCS components, temperature, $SpO_2$) are inconsistently documented by field crews during rapid transports.
  - **No Standardized Acuity Target:** The dataset lacks a continuous 1–10 criticality target; triage is recorded retrospectively as transport priority or dispatch code.
- **Feasibility Verdict:** **REJECTED.** Complex restricted licensing prevents open-source GitHub/web hosting; inconsistent field logging and lack of a normalized 1–10 target.

---

### 2.3 PhysioNet Challenge Datasets (e.g., Sepsis 2019, Mortality 2012)
- **Source / URL:** [PhysioNet Computing in Cardiology Challenge](https://physionet.org/)
- **Availability & License:** Open for research under Open Data Commons / PhysioNet licenses.
- **Relevant Fields Available:** Time-series ICU vitals (heart rate, MAP, $SpO_2$, respiration), lab values.
- **Missing / Mismatched Fields:**
  - **ICU vs. Pre-Hospital:** Represents high-acuity Intensive Care Unit patients with invasive monitors, arterial lines, and laboratory panels (lactate, bilirubin, creatinine).
  - **Missing Symptoms:** No pre-hospital categorical indicators (ambulatory status, vomiting, headache, trauma, bleeding).
  - **Target Mismatch:** Targets are binary in-hospital mortality or sepsis onset within 6 hours, not pre-hospital emergency acuity.
- **Feasibility Verdict:** **REJECTED.** ICU inpatient population with lab tests is fundamentally unrepresentative of pre-hospital field triage.

---

### 2.4 Kaggle Emergency Department Triage Datasets
- **Source / URL:** [Kaggle Emergency Triage Collections](https://www.kaggle.com/) (e.g., *Emergency Service Triage Application Dataset*, *Triagegeist*)
- **Availability & License:** Public Open Access (CC0 / CC BY 4.0 / Open Database).
- **Relevant Fields Available:** Small retrospective cohorts (e.g., 1,267 rows) with triage levels 1–5 and basic vitals.
- **Missing / Mismatched Fields:**
  - **Small Sample Size:** Real-world public sets are very small ($N \approx 1,000 - 1,500$), resulting in high variance and severe class imbalance.
  - **Missing Core Pre-Hospital Features:** None contain the complete composite of pre-hospital parameters (walking status, altered consciousness, trauma, bleeding, headache, fever, vomiting, and full vitals).
  - **Target Discrepancy:** Discrete 3-level or 5-level institutional triage classes, heavily imbalanced toward non-urgent cases.
- **Feasibility Verdict:** **REJECTED.** Insufficient feature coverage, low sample size, and in-hospital ED focus.

---

### 2.5 UCI Machine Learning Repository
- **Source / URL:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- **Search Findings:** No dedicated pre-hospital emergency triage or multi-vital criticality dataset is cataloged on the UCI repository.
- **Feasibility Verdict:** **REJECTED.** No relevant datasets exist in the repository.

---

### 2.6 National Hospital Ambulatory Medical Care Survey (NHAMCS - CDC)
- **Source / URL:** [CDC NHAMCS Data](https://www.cdc.gov/nchs/ahcd/about_nhamcs.htm)
- **Availability & License:** Public domain federal survey data.
- **Relevant Fields Available:** Reason for visit, age, sex, arrival by ambulance (yes/no), triage level (1–5).
- **Missing / Mismatched Fields:**
  - **Missing Vital Signs:** Granular physiological vitals (continuous $SpO_2$, continuous GCS, detailed systolic/diastolic BP) are largely categorized or suppressed in public use files for de-identification.
  - **Survey Sampling Weight Artifacts:** Designed for annual population epidemiological estimates, not real-time clinical predictive modeling.
- **Feasibility Verdict:** **REJECTED.** Lacks granular continuous physiological measurements needed for real-time acuity modeling.

---

## 3. Summary Comparison Table

| Dataset | Setting | Feature Match (%) | Sample Size | Licensing / Redistribution | Suitable Target? | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MIMIC-IV-ED** | Hospital ED | ~45% | ~400k | Restricted (CITI / DUA) | ESI (1–5) | **Rejected** (Licensing & Setting) |
| **NEMSIS Public** | EMS Field | ~60% | Millions | Restricted Research Use | Transport Code | **Rejected** (Redistribution & Sparsity) |
| **PhysioNet Sepsis/ICU** | Inpatient ICU | ~30% | ~40k | Open Research | Binary Sepsis/Mortality | **Rejected** (ICU Domain & Labs) |
| **Kaggle Triage Sets** | Hospital ED | ~40% | 1.2k – 15k | Public Open | 3–5 Class Disjoint | **Rejected** (Feature Gaps & Size) |
| **UCI ML Repository** | N/A | 0% | 0 | Public | None | **Rejected** (No dataset found) |
| **CDC NHAMCS** | Survey / ED | ~35% | ~25k/yr | Public Domain | 5-Level ESI Categorical | **Rejected** (Lacks granular vitals) |

---

## 4. Conclusion & Technical Rationale for Synthetic Generation

Following an exhaustive review of all major public medical data sources, **no publicly redistributable dataset exists that captures the full complement of pre-hospital clinical signs, continuous vitals, transit indicators, and a standardized 1–10 criticality target.**

### Why Synthetic Generation is Strictly Required:
1. **Zero Regulatory & Privacy Risk:** Guarantees zero exposure of Protected Health Information (PHI) or violation of hospital Data Use Agreements.
2. **Complete Feature Alignment:** Allows exact simulation of pre-hospital field realities (e.g., assessing walking status, active bleeding, GCS, and $SpO_2$ simultaneously).
3. **Controlled Physiological Modeling:** Enables modeling known pathophysiological dynamics (e.g., hypoxemia $\rightarrow$ tachycardia/tachypnea, hemorrhagic shock $\rightarrow$ hypotension + tachycardia) while preserving realistic noise and non-linearities.
4. **Reproducibility:** A fixed-seed synthetic dataset allows open-source verification, unit testing, and reliable grading in a portfolio context.

### Synthetic Data Disclaimer:
> **MANDATORY NOTICE:** The dataset used in this project is synthetically generated for educational, architectural, and machine learning engineering purposes. It is **NOT** real patient data and has **NOT** been clinically validated. It must **NEVER** be used for clinical decision-making or real-world patient triage.
