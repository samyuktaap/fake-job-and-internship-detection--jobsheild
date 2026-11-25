# JobShield: Comprehensive Project & Architecture Audit

**Role:** Senior Machine Learning Engineer  
**Date:** November 25, 2026  
**Repository:** [fake-job-and-internship-detection--jobsheild](https://github.com/samyuktaap/fake-job-and-internship-detection--jobsheild)  
**Audit Purpose:** Comprehensive pre-upgrade architecture, code, and ML pipeline audit prior to developing JobShield 2.0.

---

## 1. Executive Summary & Audit Verdict

This audit reviews the current state of **JobShield – Fake Job and Internship Detection Platform**. The platform currently functions as an NLP-assisted decision-support system designed to identify fraudulent online employment advertisements.

### Key Audit Findings:
1. **Working Baseline Status:** The current codebase is operational and passes **12/12 automated unit and integration tests** in `pytest` (execution time: 4.13s).
2. **ML Pipeline Soundness:** Preprocessing and feature transformers are fitted **strictly on training folds** inside scikit-learn `Pipeline` objects. The held-out test split of 2,378 samples was preserved without leakage during vectorizer fitting.
3. **Probability Calibration:** The production champion model uses `CalibratedClassifierCV` with Platt scaling over `LinearSVC(class_weight="balanced")`, achieving a low Brier score loss of **0.01188** and an optimal decision threshold of **0.33** tuned on validation data.
4. **Data Leakage Discovery (Near-Duplicate Descriptions):** While 2,031 exact 3-field template duplicates (`title`, `company_profile`, `description`) were removed prior to splitting, **154 job description bodies still overlap between `train.csv` and `test.csv`**. On this overlapping test subset ($N=202$, $16$ fraud cases), the model achieved an artificial **100% recall**, which inflated the aggregate test recall from **76.92%** (on strictly disjoint data) to **80.37%**.
5. **False Negative Reality:** On the held-out test split, the model misses 21 out of 107 scam postings, representing a **19.63% Fraud-Class False Negative Rate** (miss rate). All 21 missed scams represent sophisticated **enterprise impersonation / profile plagiarism**, which bag-of-words and linear models cannot catch without external domain verification.
6. **Architectural Gaps:** The platform lacks semantic sentence embeddings, recruiter domain age verification, API rate-limiting, and an automated continuous retraining loop.

---

## 2. System Architecture & Component Inspection

The JobShield repository comprises five primary subsystems:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        JobShield Architecture                          │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. Machine Learning Core       │ 2. Backend API Service                │
│    - Scikit-learn Pipeline     │    - FastAPI Service (main.py)        │
│    - TF-IDF (12,000 features)  │    - SQLite DB (data/jobshield.db)    │
│    - Categorical Imputer / OHE │    - Pydantic v2 Schemas              │
│    - Domain Risk Extractor     │    - Endpoints: predict, batch, etc.  │
│    - Calibrated Linear SVM     │                                       │
├────────────────────────────────┼───────────────────────────────────────┤
│ 3. Explainability Engine       │ 4. User Frontends                     │
│    - JobShieldExplainer        │    - Web SPA (frontend/index.html)    │
│    - Coefficient Attribution   │    - Browser Extension (extension/)   │
│    - Heuristic Red/Green Flags │      (Manifest V3 local unpacked)     │
└────────────────────────────────┴───────────────────────────────────────┘
```

### 2.1 Backend Architecture (`backend/`)
- **Framework:** FastAPI (`0.141.1`), served via Uvicorn.
- **Service Endpoints (`backend/main.py`):**
  - `GET /api/health`: Health status, model loaded flag, active decision threshold (`0.33`).
  - `POST /api/predict`: Validates payload via `JobAnalysisRequest`, executes inference and explanation, logs record to SQLite, and returns calibrated probability, risk level, heuristic red flags, and feature importances.
  - `POST /api/predict/batch`: Accepts up to 50 job postings in a single request for batch processing.
  - `GET /api/history`: Returns historical job scans from the SQLite database.
  - `POST /api/feedback`: Accepts candidate/reviewer feedback regarding false positives or false negatives.
  - `GET /api/metrics`: Exposes model benchmark comparisons from serialized artifacts.
  - `GET /api/samples`: Delivers 4 curated demo test cases (2 legitimate, 2 fraudulent).
  - `GET /api/error-analysis`: Returns investigated false positives and false negatives from test evaluation.
  - `GET /`: Serves static web frontend (`frontend/index.html`).
- **Data Validation (`backend/schemas.py`):** Built with Pydantic v2. Enforces non-empty strings and min length of 10 characters on `description`.
- **Database Persistence (`backend/database.py`):** SQLite database located at `data/jobshield.db`. Tables: `scan_history` (historical predictions) and `user_feedback` (user corrections).

### 2.2 Machine Learning Core (`src/`)
- **Text Preprocessing (`src/features/preprocessor.py`):**
  - `JobTextExtractor`: Cleans HTML tags, strips URLs, normalizes whitespace, and concatenates text fields with weighted title (`title` $\times 2$, `location`, `department`, `company_profile`, `description`, `requirements`, `benefits`).
  - `TfidfVectorizer`: Unigrams and bigrams (`ngram_range=(1, 2)`), `max_features=12000`, `sublinear_tf=True`, English stop words, `min_df=3`, `max_df=0.9`.
- **Domain Heuristic Indicators (`src/features/risk_indicators.py`):**
  - `DomainRiskExtractor`: Extracted 10 binary tabular features:
    1. `risk_payment_mention`: Wire transfer, Western Union, MoneyGram, upfront fees, crypto, cashier's checks.
    2. `risk_free_email`: Mentions of `@gmail.com`, `@yahoo.com`, `@hotmail.com`, etc.
    3. `risk_unrealistic_earning`: Promises of thousands weekly or daily cash without qualifications.
    4. `risk_urgency_language`: "Immediate start", "act fast", "urgent hiring".
    5. `risk_missing_company_profile`: Company profile string length $< 30$ chars.
    6. `risk_no_company_logo`: `has_company_logo == 0`.
    7. `risk_telecommuting_flag`: Remote position indicator.
    8. `risk_no_screening_questions`: `has_questions == 0`.
    9. `risk_missing_requirements`: Requirements string length $< 15$ chars.
    10. `risk_brief_description`: Description string length $< 100$ chars.
- **Categorical Preprocessing:** One-hot encoding on `employment_type`, `required_experience`, and `required_education` with `handle_unknown="ignore"`.
- **Pipeline Architecture:** Scikit-learn `FeatureUnion` combining the TF-IDF pipeline, categorical pipeline, and domain risk pipeline into a single unified matrix.

### 2.3 Explainability Engine (`src/explainability/explainer.py`)
- **Transparency Method:** `JobShieldExplainer` generates three distinct explainability outputs:
  1. *Calibrated Risk Probability:* Model output mapped to 4 risk tiers (Low $<20\%$, Moderate $20-33\%$, Elevated $33-70\%$, High $\ge 70\%$).
  2. *Heuristic Flags:* Regex-driven red flags and positive legitimacy markers.
  3. *Linear Feature Attribution:* Extracts weights from the calibrated linear SVM ensemble:
     $$\bar{w} = \frac{1}{K} \sum_{k=1}^K w_k$$
     Computes local feature contributions $c_j = \bar{w}_j \cdot x_j$ for active features.
  4. *Responsible-Use Notice:* Mandatory disclaimer stating predictions are screening aids, not legal determinations.

### 2.4 Browser Extension (`extension/`)
- **Manifest Version:** Manifest V3 (`manifest.json`).
- **Components:**
  - `popup.html` & `popup.js`: Extension popup UI showing API connectivity status and scan trigger.
  - `content.js`: Injected into LinkedIn (`*.linkedin.com/*`) and Indeed (`*.indeed.com/*`) to extract `title`, `company`, `description`, and metadata from active DOM nodes.
  - `background.js`: Service worker managing background messaging.
- **Operational Status:** **Developer Prototype.** Installed via `chrome://extensions` ("Load unpacked"). Communicates with local host `http://127.0.0.1:8000`.

---

## 3. Automated Test Suite Audit

The test suite was executed in PowerShell using Python 3.12.10:

```
python -m pytest tests -v
```

### Test Results Breakdown:
| Test File | Test Identifier | Purpose | Result |
| :--- | :--- | :--- | :---: |
| `tests/test_api.py` | `test_health_endpoint` | Health endpoint status and threshold verification | **PASSED** |
| `tests/test_api.py` | `test_metrics_endpoint` | Model comparison metadata delivery | **PASSED** |
| `tests/test_api.py` | `test_samples_endpoint` | Delivery of 4 curated benchmark samples | **PASSED** |
| `tests/test_api.py` | `test_predict_endpoint_valid_and_persisted` | Prediction schema and SQLite scan persistence | **PASSED** |
| `tests/test_api.py` | `test_predict_endpoint_fraud_case` | Phishing sample classification and red flags | **PASSED** |
| `tests/test_api.py` | `test_predict_endpoint_invalid_short_description` | Pydantic 422 validation on short text | **PASSED** |
| `tests/test_api.py` | `test_batch_predict_endpoint` | Batch API processing multiple listings | **PASSED** |
| `tests/test_api.py` | `test_history_endpoint` | SQLite scan history retrieval | **PASSED** |
| `tests/test_api.py` | `test_feedback_endpoint` | SQLite user feedback logging | **PASSED** |
| `tests/test_pipeline.py` | `test_clean_text` | HTML sanitization and whitespace normalization | **PASSED** |
| `tests/test_pipeline.py` | `test_risk_extractor` | Extraction of 10 domain risk features | **PASSED** |
| `tests/test_pipeline.py` | `test_explainer_prediction` | End-to-end model inference and explanation parity | **PASSED** |

**Summary: 12 passed, 0 failed in 4.13 seconds (100% pass rate).**

---

## 4. Feature Classification: Implemented vs. Prototype vs. Missing

| System Component | Feature | Current Status | Verification Source |
| :--- | :--- | :---: | :--- |
| **ML Core** | TF-IDF + Domain Risk FeatureUnion | ✅ Implemented | `src/features/preprocessor.py` |
| **ML Core** | 6 Model Benchmark & 5-Fold CV | ✅ Implemented | `artifacts/audit_metrics.json` |
| **ML Core** | Calibrated Linear SVM Champion | ✅ Implemented | `artifacts/jobshield_pipeline.joblib` |
| **ML Core** | Validation Threshold Tuning ($0.33$) | ✅ Implemented | `artifacts/champion_metadata.json` |
| **ML Core** | Dense Transformer Embeddings | ❌ Missing | Not in pipeline; planned for v2.0 |
| **ML Core** | Ensemble Soft-Voting Stacker | ❌ Missing | Single model in production |
| **Explainability** | Heuristic Red & Green Flag Rules | ✅ Implemented | `src/explainability/explainer.py` |
| **Explainability** | Linear Fold Coefficient Attribution | ✅ Implemented | `src/explainability/explainer.py` |
| **Explainability** | Contextual SHAP/Attention Heatmaps | ❌ Missing | Not implemented |
| **Backend** | Single Job Prediction API | ✅ Implemented | `backend/main.py:POST /api/predict` |
| **Backend** | Batch Job Prediction API | ✅ Implemented | `backend/main.py:POST /api/predict/batch` |
| **Backend** | SQLite Database Persistence | ✅ Implemented | `backend/database.py` (`data/jobshield.db`) |
| **Backend** | User Feedback & Correction API | ✅ Implemented | `backend/main.py:POST /api/feedback` |
| **Backend** | API Rate Limiting (`slowapi`) | ❌ Missing | No rate limiting middleware |
| **Backend** | API Authentication (API Keys/JWT) | ❌ Missing | Public open endpoints |
| **Frontend** | Single-Page Web App Dashboard | ✅ Implemented | `frontend/index.html` |
| **Frontend** | Model Benchmark Matrix & Visuals | ✅ Implemented | `frontend/index.html` (Tab 2) |
| **Frontend** | Scan History Drawer | ✅ Implemented | `frontend/index.html` (Tab 4) |
| **Extension** | Manifest V3 Extension | ⚠️ Prototype | `extension/` (Requires local server) |
| **Extension** | DOM Scrapers for LinkedIn & Indeed | ⚠️ Prototype | Tested locally; fragile to DOM changes |
| **Extension** | Chrome Web Store Publication | ❌ Missing | Developer unpacked build only |
| **Security** | Recruiter Domain & WHOIS Verification | ❌ Missing | No external DNS/WHOIS checking |

---

## 5. Highest-Priority Issues & Blockers to Fix Before Upgrade

Before proceeding with JobShield 2.0 implementation, four critical architectural blockers must be addressed:

### Issue 1: Near-Duplicate Description Leakage in Held-Out Test Set
- **Problem:** Deduplication on `(title, company_profile, description)` allowed 154 duplicate description strings to bridge the train and test splits (376 train rows, 202 test rows, with 16 fraud cases in test).
- **Impact:** The model achieved 100% recall on the overlapping test fraud cases, artificially raising reported test recall from **76.92%** to **80.37%**.
- **Remedy:** Re-partition the held-out test split using a strict **content-hash / MinHash LSH deduplication** strategy across the job description body so that test evaluation reflects true zero-shot generalization.

### Issue 2: Enterprise Impersonation Blind Spot (21 False Negatives)
- **Problem:** All 21 missed fraud cases in the test set copied authentic corporate job listings from legitimate enterprises (e.g., real estate, engineering, healthcare). The scam was orchestrated post-application via private email or phone.
- **Impact:** Pure NLP bag-of-words cannot detect a scam when the scammer uses authentic corporate prose.
- **Remedy:** Implement an external **domain alignment module**: compare recruiter contact emails and redirect URLs against official corporate domains and check domain registration age via WHOIS.

### Issue 3: Inability to Handle Paraphrased or LLM-Rewritten Scams
- **Problem:** TF-IDF with 12,000 features relies on exact lexical n-gram matches. Modern scams written with LLMs or novel phrasing bypass static keyword lists.
- **Impact:** Model vulnerability to adversarial paraphrasing and lexical drift.
- **Remedy:** Integrate dense semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` in a hybrid representation.

### Issue 4: Backend Security & Rate Limiting Gap
- **Problem:** `/api/predict` and `/api/predict/batch` have no rate limiting (`slowapi`) or authentication.
- **Impact:** High vulnerability to denial-of-service or scraping attacks.
- **Remedy:** Integrate `slowapi` rate limiting and optional API key middleware.

---

## 6. Actionable Implementation Roadmap: JobShield 2.0

```
┌────────────────────────────────────────────────────────────────────────┐
│                      JobShield 2.0 Upgrade Plan                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
 ┌─────────────────────────────────▼──────────────────────────────────┐
 │ Phase 1: Data Pipeline Hardening & Disjoint Split                  │
 │ - MinHash LSH deduplication on description body                    │
 │ - Clean zero-leakage evaluation benchmark                          │
 └─────────────────────────────────┬──────────────────────────────────┘
                                   │
 ┌─────────────────────────────────▼──────────────────────────────────┐
 │ Phase 2: Feature Engineering & Semantic Text Representation        │
 │ - Sentence Transformers (all-MiniLM-L6-v2) embedding extraction   │
 │ - Structured heuristic extraction (salary, domain, contact flags)  │
 │ - Email/domain verification heuristics                             │
 └─────────────────────────────────┬──────────────────────────────────┘
                                   │
 ┌─────────────────────────────────▼──────────────────────────────────┐
 │ Phase 3: Hybrid Ensemble Architecture                              │
 │ - Calibrated Linear SVM (sparse lexical)                           │
 │ - XGBoost / LightGBM (dense tabular + embeddings)                  │
 │ - Calibrated Soft-Voting Meta-Learner (Platt calibrated)           │
 └─────────────────────────────────┬──────────────────────────────────┘
                                   │
 ┌─────────────────────────────────▼──────────────────────────────────┐
 │ Phase 4: Explainable AI & Transparency Upgrade                     │
 │ - Attention/Token importance highlighting                          │
 │ - Risk breakdown: Lexical vs. Semantic vs. Domain signals          │
 └─────────────────────────────────┬──────────────────────────────────┘
                                   │
 ┌─────────────────────────────────▼──────────────────────────────────┐
 │ Phase 5: Production Hardening & Extension Packaging                │
 │ - API rate limiting via slowapi                                    │
 │ - Chrome Web Store ready manifest packaging                        │
 └────────────────────────────────────────────────────────────────────┘
```

> **Senior ML Engineer Recommendation:** Do not write new model code until the data split is cleaned of near-duplicate description leakage. Proceed to review [DATASET_AUDIT.md](file:///c:/Users/Samyuktaa%20p/ml/DATASET_AUDIT.md) and [ML_BASELINE_REPORT.md](file:///c:/Users/Samyuktaa%20p/ml/ML_BASELINE_REPORT.md) before approving Phase 1.
