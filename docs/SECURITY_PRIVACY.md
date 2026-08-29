# Healthcare Security, Privacy & Compliance Architecture

## Document Information
- **Project Title:** AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
- **Document Version:** 1.0.0
- **Classification:** Clinical AI Security, HIPAA Compliance & Telemetry Data Governance

---

## 1. Executive Summary & Compliance Mandate

Pre-hospital emergency healthcare applications operate at the intersection of extreme time-sensitivity and stringent regulatory mandates. When transitioning an AI prototype into clinical emergency operations, strict compliance with the **Health Insurance Portability and Accountability Act (HIPAA)**, the **EU General Data Protection Regulation (GDPR)**, and **FDA Cybersecurity in Medical Devices Guidelines** is mandatory.

This document establishes the security architecture, encryption standards, Protected Health Information (PHI) handling protocols, and API safeguards required for pre-hospital clinical AI systems.

---

## 2. Protected Health Information (PHI) & Data Governance

### 2.1 Safe Harbor De-Identification (HIPAA § 164.514)
In this educational and portfolio prototype, **no real-world patient data is ingested or stored**. All records are generated using a fixed-seed, physiologically modeled synthetic generator.

For real-world clinical implementation, the 18 HIPAA Safe Harbor identifiers must be stripped before model training or analytical ingestion:
1. Names
2. Geographic subdivisions smaller than a state (except first 3 digits of ZIP code under conditions)
3. All elements of dates (except year) directly related to an individual (e.g., exact birth dates, admission dates)
4. Telephone numbers
5. Fax numbers
6. Electronic mail addresses
7. Social Security numbers
8. Medical record numbers (MRNs replaced with cryptographically salted pseudo-identifiers)
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers and serial numbers (including ambulance VINs)
13. Device identifiers and serial numbers
14. Web Universal Resource Locators (URLs)
15. Internet Protocol (IP) address numbers
16. Biometric identifiers (fingerprints, voiceprints)
17. Full face photographic images
18. Any other unique identifying number, characteristic, or code

---

## 3. Cryptographic Security Standards

### 3.1 Data in Transit (Ambulance Telemetry $\rightarrow$ Hospital Server)
Emergency pre-hospital ambulances transmit vital signs and triage inputs over cellular (4G/5G) or satellite networks.
- **Protocol:** Transport Layer Security (TLS 1.3 mandatory; TLS 1.2 minimum with strict cipher suites).
- **Cipher Suites:** `TLS_AES_256_GCM_SHA384`, `ECDHE-RSA-AES256-GCM-SHA384`.
- **Mutual TLS (mTLS):** Enforced between ambulance mobile field terminals and hospital edge gateways to prevent man-in-the-middle (MitM) attacks.
- **Payload Integrity:** Cryptographic HMAC-SHA256 checksums attached to each vital telemetry packet.

### 3.2 Data at Rest (Triage Database & Serialized Models)
- **Database Encryption:** AES-256-GCM encryption for all database volumes, transaction logs, and temporary buffer stores.
- **Model Integrity & Signing:** Trained serialized artifacts (`models/pipeline.joblib`) are digitally signed with an RSA-4096 private key; the inference service verifies the SHA-256 checksum upon container boot before loading into memory.
- **Key Management:** Encryption keys managed via FIPS 140-3 Level 3 compliant Hardware Security Modules (AWS KMS, Azure Key Vault, or HashiCorp Vault) with automated 90-day rotation.

---

## 4. Operational Access Control & Role-Based Security (RBAC)

| User Role | Pre-Hospital Data Input | View Real-Time Acuity | View SHAP Attributions | Manage Triage Queue | Access Model Artifacts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Field Paramedic / EMT** | ✅ Read/Write | ✅ Read | ✅ Read | ❌ Read-Only | ❌ No Access |
| **Emergency Dispatcher** | ✅ Read/Write | ✅ Read | ❌ No Access | ✅ Manage Queue | ❌ No Access |
| **ED Triage Nurse / MD** | ✅ Read/Write | ✅ Read | ✅ Read | ✅ Prioritize Bays | ❌ No Access |
| **MLOps / Clinical Engineer** | ❌ No Direct PHI | ❌ Audit Only | ✅ Audit Only | ❌ No Access | ✅ Model CI/CD |

- **Authentication:** OpenID Connect (OIDC) / OAuth 2.0 with mandatory Multi-Factor Authentication (MFA) via FIDO2/WebAuthn or hardware security keys.
- **Session Lifespans:** Field mobile sessions expire after 15 minutes of inactivity; emergency tablet PIN re-authentication required.

---

## 5. Inference API Security & Hardening

1. **Input Sanitization & Boundary Clamping:**
   - Handled via strict Pydantic schemas enforcing physiological bounds ($SpO_2 \in [50, 100]$, $HR \in [30, 250]$, $SBP \in [50, 260]$, $GCS \in [3, 15]$).
   - Prevents injection attacks, adversarial boundary perturbations, and buffer overflow attempts.
2. **Rate Limiting & DDoS Protection:**
   - Token-bucket rate limiting (e.g. max 60 requests/minute per ambulance terminal ID).
3. **Audit Logging (HIPAA § 164.312(b)):**
   - Immutable, append-only audit trail logging every prediction event:
     - Timestamp (UTC ISO 8601)
     - Pseudonymized Encounter ID
     - Authenticated Clinician User ID
     - Model Version Hash
     - Predicted Criticality Score & Urgency Tier
     - Safety Override Status (Red flags triggered)
   - Logs stored in Write-Once-Read-Many (WORM) cloud storage with 7-year retention.

---

## 6. Emergency Offline Fallback Protocol

In disaster zones or transit through cellular dead-zones, the system must degrade safely:
1. **Edge Container Fallback:** The pre-hospital tablet runs a local lightweight containerized inference service with cached baseline pipelines.
2. **Deterministic Hard Safety Rules:** If the ML runtime encounters a memory or hardware fault, the application falls back immediately to deterministic heuristic triage protocols (e.g., standard START / NEWS2 threshold table).
3. **Store-and-Forward Telemetry:** Once cellular connectivity is restored, cached encrypted encounter logs are securely synchronized with the hospital emergency receiving server.
