# Feature Dictionary & Data Schema

## Document Information
- **Project Title:** AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
- **Document Version:** 1.0.0
- **Status:** Phase 1 Completed / Schema Defined
- **Target Population:** Simulated Pre-Hospital Adult & Adolescent Emergency Patients ($N = 10,000$)

---

## 1. Identifier Field (Non-Predictive)

| Attribute | Specification |
| :--- | :--- |
| **Feature Name** | `patient_id` |
| **Data Type** | `string` / `object` |
| **Meaning** | Synthetic alphanumeric record identifier (e.g., `PT-10001` to `PT-20000`). |
| **Possible Values** | Unique string per simulated patient. |
| **Why It Matters** | Ensures tracking, audit logging, and referencing of individual records in testing. |
| **Used by Model?** | **NO.** Explicitly dropped during preprocessing to prevent identity leakage and spurious memorization. |
| **Available Pre-Hospital?** | Yes (assigned by dispatch or mobile field unit). |
| **Generation Method** | Sequential integer formatted with prefix `PT-`. |

---

## 2. Demographic & Transit Features

| Attribute | `age` | `sex` | `ambulance_arrival` |
| :--- | :--- | :--- | :--- |
| **Data Type** | `float64` / `int64` | `string` / `category` | `int64` (Binary 0/1) |
| **Meaning** | Age of the patient in years. | Biological sex of the patient. | Mode of transit to the emergency center. |
| **Possible Values** | $16 - 95$ years | `Male`, `Female`, `Other` | `0` (Walk-in / Private Vehicle), `1` (EMS / Ambulance) |
| **Why It Matters** | Extremes of age (elderly $\ge 70$) exhibit reduced physiological reserve and higher mortality risks. | Contextual demographic baseline; slight difference in baseline vital distributions. | Ambulance transit strongly correlates with acute field dispatch, trauma, or severe illness. |
| **Used by Model?** | **YES** | **YES** (One-Hot Encoded) | **YES** |
| **Available Pre-Hospital?** | Yes (reported by patient, family, or bystander). | Yes (observable or reported). | Yes (inherent to the pre-hospital context). |
| **Generation Method** | Bimodal distribution reflecting general emergency and geriatric cohorts. | Categorical sampling: $50\%$ Male, $48\%$ Female, $2\%$ Other. | Bernoulli sampling conditioned on latent severity ($P(\text{Amb}) \approx 0.25$ for mild, $\approx 0.85$ for critical). |

---

## 3. Observable Symptoms & Physical Signs

| Feature Name | Data Type | Possible Values | Why It Matters (Clinical Relevance) | Used by Model? | Available Pre-Hospital? | Generation Logic |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| `walking_ability` | `int64` (Binary) | `0` (No), `1` (Yes) | Inability to walk (non-ambulatory) is the primary criterion in START triage indicating compromised stability. | **YES** | Yes | Sampled with inverse probability relative to latent severity score. |
| `altered_consciousness` | `int64` (Binary) | `0` (No), `1` (Yes) | Confusion, lethargy, or unresponsiveness indicates cerebral hypoperfusion, stroke, intoxication, or severe hypoxia. | **YES** | Yes | High probability when GCS $< 14$ or latent neurological/metabolic crisis is present. |
| `chest_pain` | `int64` (Binary) | `0` (No), `1` (Yes) | Major red-flag symptom for Acute Coronary Syndrome (ACS), pulmonary embolism, or aortic dissection. | **YES** | Yes | Bernoulli trial influenced by age, cardiac history, and cardiac distress sub-factor. |
| `difficulty_breathing` | `int64` (Binary) | `0` (No), `1` (Yes) | Dyspnea indicates respiratory failure, acute asthma, COPD exacerbation, or heart failure. | **YES** | Yes | Strongly linked to lower $SpO_2$ and elevated respiratory rate. |
| `abdominal_pain` | `int64` (Binary) | `0` (No), `1` (Yes) | Red-flag for acute abdomen, appendicitis, internal hemorrhage, bowel obstruction, or pancreatitis. | **YES** | Yes | Moderate baseline prevalence ($22\%$) with variable acuity weights. |
| `injury_trauma` | `int64` (Binary) | `0` (No), `1` (Yes) | Identifies mechanical impact, motor vehicle collisions, falls, or penetrating wounds. | **YES** | Yes | Independent trauma risk factor, correlated with active bleeding and tachycardia. |
| `bleeding` | `int64` (Binary) | `0` (No), `1` (Yes) | External or observable internal hemorrhage threatening hypovolemic shock. | **YES** | Yes | Strongly correlated with low systolic BP, elevated pulse, and trauma flags. |
| `fever` | `int64` (Binary) | `0` (No), `1` (Yes) | Subjective report or tactile indicator of systemic infection or sepsis. | **YES** | Yes | Correlated with elevated body temperature ($\ge 38.0^\circ\text{C}$). |
| `headache` | `int64` (Binary) | `0` (No), `1` (Yes) | Symptom of hypertension, intracranial pathology, infection, or tension. | **YES** | Yes | Sampled across low-to-moderate acuity cohorts, with higher criticality if hypertensive. |
| `vomiting` | `int64` (Binary) | `0` (No), `1` (Yes) | Risk factor for dehydration, electrolyte imbalance, intracranial pressure, or poisoning. | **YES** | Yes | Sampled across acute abdominal, metabolic, or systemic complaints. |

---

## 4. Field Physiological Vital Signs

| Feature Name | Data Type | Clinical Normal Range | Simulated Range | Why It Matters (Physiological Relevance) | Used by Model? | Generation Logic |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| `heart_rate` | `float64` (bpm) | $60 - 100$ | $38 - 195$ | Tachycardia ($>100$) indicates pain, shock, sepsis, or arrhythmia; Bradycardia ($<50$) indicates heart block or pre-terminal collapse. | **YES** | Gaussian distribution shifted by shock, respiratory distress, and fever states. |
| `systolic_bp` | `float64` (mmHg) | $100 - 130$ | $60 - 235$ | Hypotension ($<90$) defines shock states; Extreme hypertension ($>190$) risks hypertensive crisis / stroke. | **YES** | Correlated with diastolic BP; depressed by hemorrhage/sepsis; elevated by distress/pain. |
| `diastolic_bp` | `float64` (mmHg) | $60 - 85$ | $35 - 135$ | Reflects systemic vascular resistance and coronary perfusion pressure. | **YES** | Generated via physiological pulse pressure offset ($SBP - DBP \approx 30 - 60$ mmHg). |
| `spo2` | `float64` (%) | $95 - 100$ | $68 - 100$ | Hypoxemia ($<92\%$) indicates respiratory failure and tissue hypoxia; primary emergency triage factor. | **YES** | Beta/Truncated Normal distribution decaying rapidly in high respiratory distress. |
| `respiratory_rate` | `float64` (breaths/min) | $12 - 20$ | $8 - 48$ | Tachypnea ($>24$) is the most sensitive early marker of physiological deterioration (sepsis, metabolic acidosis). | **YES** | Inverse correlation with $SpO_2$; shifted higher by fever, shock, and pain. |
| `temperature` | `float64` (°C) | $36.5 - 37.5$ | $34.2 - 41.2$ | Hyperthermia ($>38.5^\circ\text{C}$) suggests severe infection; Hypothermia ($<35.5^\circ\text{C}$) suggests environmental exposure or severe septic shock. | **YES** | Gaussian around $37.0^\circ\text{C}$ with right-skew for fever cases and lower tail for septic/trauma shock. |
| `gcs` | `int64` (Score) | $15$ | $3 - 15$ | Glasgow Coma Scale measures neurological responsiveness (Eye, Verbal, Motor). $GCS \le 8$ defines coma/airway risk. | **YES** | Discrete ordinal scoring conditioned on altered consciousness, trauma, and hypoxemia. |

---

## 5. Justified Additional Pre-Hospital Context Features

These 5 features are added because they are standard, non-invasive, pre-hospital parameters that significantly improve clinical realism without requiring laboratory testing:

| Feature Name | Data Type | Possible Values | Why It Matters (Clinical Justification) | Used by Model? | Generation Logic |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `pain_severity` | `int64` (0–10 NRS) | $0 - 10$ | Standard vital parameter in triage; severe acute pain ($8–10$) alters hemodynamics (HR, BP) and impacts urgency. | **YES** | Sampled from 0–10 conditioned on trauma, chest pain, and abdominal pain flags. |
| `oxygen_requirement` | `int64` (Binary) | `0` (No), `1` (Yes) | Field EMS administration of supplemental oxygen directly indicates clinical respiratory insufficiency. | **YES** | Strongly conditioned on $SpO_2 < 93\%$ and difficulty breathing. |
| `known_cardiac_history` | `int64` (Binary) | `0` (No/Unk), `1` (Yes) | History of MI/CHF markedly elevates the pre-test probability of acute cardiac decompensation. | **YES** | Sampled with higher prevalence in older cohorts ($age \ge 55$). |
| `known_hypertension` | `int64` (Binary) | `0` (No/Unk), `1` (Yes) | Informs interpretation of elevated systolic/diastolic blood pressure measurements. | **YES** | Age-dependent prevalence ($30–60\%$). |
| `known_diabetes` | `int64` (Binary) | `0` (No/Unk), `1` (Yes) | Diabetic patients often present with atypical symptoms (e.g., painless MI) and severe metabolic derangements. | **YES** | Sampled across cohort ($15–25\%$). |

---

## 6. Summary of Feature Counts

- **Total Generated Columns:** 21 predictive features + 1 non-predictive identifier (`patient_id`) + 2 target labels (`criticality_score`, `urgency_category`).
- **Numerical Features (8):** `age`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `spo2`, `respiratory_rate`, `temperature`, `pain_severity`.
- **Discrete Ordinal Features (1):** `gcs` (can be treated as numerical in scaling/modeling).
- **Binary Indicator Features (12):** `ambulance_arrival`, `walking_ability`, `altered_consciousness`, `chest_pain`, `difficulty_breathing`, `abdominal_pain`, `injury_trauma`, `bleeding`, `fever`, `headache`, `vomiting`, `oxygen_requirement`, `known_cardiac_history`, `known_hypertension`, `known_diabetes`.
- **Categorical Features (1):** `sex` (`Male`, `Female`, `Other`).
