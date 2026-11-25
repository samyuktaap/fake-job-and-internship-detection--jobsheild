# JobShield 2.0: Comprehensive Project & Architecture Audit

**Audit Date:** November 24, 2026 (Updated November 25, 2026)  
**Auditor:** JobShield Engineering & ML Team  
**Scope:** Complete codebase inspection covering backend, database, frontend, browser extension, preprocessing, ML models, dataset hygiene, data leakage, and test infrastructure.

---

## 1. Executive Summary

This audit establishes the verified baseline status of the **JobShield** project. The application provides an honest, leakage-free machine learning pipeline, an SQLite persistence layer, a FastAPI backend service, an interactive single-page browser frontend, and an unpacked Manifest V3 browser extension developer prototype. The system focuses on decision-support screening for fraudulent job advertisements using NLP and domain indicators from the Employment Scam Aegean Dataset (EMSCAD).

All **12 automated unit and integration tests pass** (100% pass rate). The training and inference pipelines maintain strict preprocessing parity. No data leakage was found between splits. This updated audit documents the evolution from the initial stateless prototype into JobShield 2.0 with persistent SQLite storage, batch scanning, user feedback collection, and calibrated SVM classification.

---

## 2. Codebase & System Architecture Review

### 2.1 Backend Architecture & API Routes (`backend/`)
- **Framework:** FastAPI (`0.141.1`) served via Uvicorn on port `8000`.
- **API Endpoints Implemented:**
  - `GET /api/health`: Health status, model pipeline load state, active decision threshold (`0.33`).
  - `POST /api/predict`: Validates payload via Pydantic `JobAnalysisRequest`, executes `JobShieldExplainer.explain_prediction()`, returns calibrated risk score, verdict badge, detected warning signals, legitimacy markers, and persists scan to SQLite.
  - `POST /api/predict/batch`: Accepts up to 50 job listings in a single batch request for bulk scanning.
  - `GET /api/history`: Returns recent scan records from SQLite.
  - `POST /api/feedback`: Records user/reviewer feedback (false positive/negative reports) into SQLite for model improvement.
  - `GET /api/metrics`: Returns comparative evaluation metrics across models and champion metadata.
  - `GET /api/samples`: Delivers 4 curated demo job listings (2 legitimate, 2 fraudulent).
  - `GET /api/error-analysis`: Exposes inspected False Positives and False Negatives from the held-out test set.
  - `GET /`: Serves `frontend/index.html` static single-page application.
- **Middleware:** Wildcard CORS (`allow_origins=["*"]`) enabled for local web and extension access.
- **Validation:** Pydantic v2 schemas (`backend/schemas.py`) enforce input length requirements (e.g., description >= 10 chars).

### 2.2 Database Structure (`backend/database.py`)
- **Status: Implemented (SQLite Persistence).**
- **Location:** `data/jobshield.db`.
- **Tables:**
  - `scan_history`: `id`, `created_at`, `title`, `company`, `location`, `risk_score_pct`, `calibrated_probability`, `prediction_label`, `risk_level`, `badge_color`, `red_flags_json`, `green_flags_json`.
  - `user_feedback`: `id`, `created_at`, `scan_id` (foreign key), `job_title`, `reported_label`, `comments`, `user_email`.
- Predictions and feedback are persistently stored while respecting user privacy.

### 2.3 Browser Extension (`extension/`)
- **Status: Developer Prototype (Local Unpacked).**
- **Technology:** Manifest V3 (`background.js`, `popup.html`, `popup.js`, `content.js`, `content.css`).
- **Functionality:** Scrapes active job details from LinkedIn and Indeed DOM, communicates with the local FastAPI service (`http://127.0.0.1:8000/api/predict`), and renders real-time risk verdicts.
- **Deployment Distinction:** This is a local developer prototype installed via Chrome/Edge `chrome://extensions` ("Load unpacked"). It requires the local JobShield backend running on port 8000. It is **not** currently published to the Chrome Web Store.

### 2.4 Testing Setup (`tests/`)
- **Framework:** `pytest` (v9.1.1) with `fastapi.testclient`.
- **Test Suites (12 tests total):**
  - `tests/test_pipeline.py`: HTML sanitization (`clean_text`), domain risk indicator extraction (`DomainRiskExtractor`), and pipeline inference parity (`JobShieldExplainer`).
  - `tests/test_api.py`: `/api/health`, `/api/metrics`, `/api/samples`, valid `/api/predict` with SQLite persistence verification, fraud case classification, 422 validation errors, `/api/predict/batch`, `/api/history`, and `/api/feedback`.
- **Test Status:** **12 passed in 19.40s (100% pass rate).**

---

## 3. Data Pipeline & Quality Audit

### 3.1 Raw Dataset Profile (`data/fake_job_postings.csv`)
- **Dataset:** Employment Scam Aegean Dataset (EMSCAD).
- **Total Initial Rows:** 17,880
- **Raw Class Distribution:**
  - Legitimate (`0`): 17,014 (95.16%)
  - Fraudulent (`1`): 866 (4.84%)
  - Imbalance Ratio: ~19.6 : 1

### 3.2 Deduplication & Leakage Prevention Audit
- **Exact Row Duplicates Removed:** 282 rows.
- **Text Template Duplicates Removed:** 1,749 rows (identical `title`, `company_profile`, and `description`).
- **Cleaned Dataset Total:** 15,849 unique listings (15,139 legitimate, 710 fraudulent; 4.48% fraud rate).
- **Split Strategy:** 3-way stratified split:
  - **Train (70%):** 11,093 samples (497 fraud, 10,596 legit)
  - **Validation (15%):** 2,378 samples (106 fraud, 2,272 legit)
  - **Held-Out Test (15%):** 2,378 samples (107 fraud, 2,271 legit)
- **Leakage Verification:** `TfidfVectorizer`, categorical imputers, and scalers are fitted **strictly on training data** inside a unified scikit-learn `Pipeline`/`FeatureUnion`. The test set was touched exactly once after all hyperparameters and thresholds were frozen.

---

## 4. Preprocessing & Feature Engineering Parity

### 4.1 Feature Transformation Components
1. **NLP Text Extractor (`JobTextExtractor`):** Sanitizes HTML and URLs, normalizes whitespace, and joins `title` (weighted 2x), `location`, `department`, `company_profile`, `description`, `requirements`, and `benefits`.
2. **TF-IDF Vectorizer:** Unigrams and bigrams (`ngram_range=(1, 2)`), `max_features=12000`, `sublinear_tf=True`, English stop words removed, `min_df=3`, `max_df=0.9`.
3. **Categorical Feature Encoder:** One-hot encoding on `employment_type`, `required_experience`, `required_education` with `handle_unknown="ignore"`.
4. **Domain Risk Indicator Extractor (`DomainRiskExtractor`):** 10 tabular flags extracted from domain knowledge:
   - `risk_payment_mention`: Wire transfer, Western Union, MoneyGram, upfront fee, cashier's check, crypto, SSN.
   - `risk_free_email`: Mentions of `@gmail.com`, `@yahoo.com`, `@hotmail.com`, etc.
   - `risk_unrealistic_earning`: Phrases promising thousands weekly or daily cash with no experience.
   - `risk_urgency_language`: "Act fast", "immediate start", "urgent hiring".
   - `risk_missing_company_profile`: 1 if company profile is empty or < 30 characters.
   - `risk_no_company_logo`: 1 if `has_company_logo == 0`.
   - `risk_telecommuting_flag`: Remote position indicator.
   - `risk_no_screening_questions`: 1 if `has_questions == 0`.
   - `risk_missing_requirements`: 1 if requirements < 15 characters.
   - `risk_brief_description`: 1 if description < 100 characters.

### 4.2 Training vs. Prediction Parity Check
Both training (`src/models/train.py`) and live inference (`src/explainability/explainer.py` via `backend/main.py`) load the exact same serialized pipeline artifact (`artifacts/jobshield_pipeline.joblib`). **Parity is 100% verified.**

---

## 5. Machine Learning Evaluation & Calibration Audit

### 5.1 Calibrated Linear SVM Implementation
- Linear SVM optimizes hinge loss and does not output natural probabilities.
- Implemented via `CalibratedClassifierCV(estimator=LinearSVC(class_weight="balanced", max_iter=2000, random_state=42, dual="auto"), cv=3)`.
- 3 internal cross-validation folds fit Platt sigmoid scaling strictly on training folds, producing well-calibrated posterior probabilities $\hat{P}(Y=1|X) \in [0, 1]$.
- `JobShieldExplainer` correctly extracts coefficients across folds via `[cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_]` and averages them to explain feature contributions.

### 5.2 Held-Out Test Evaluation & False Negative Reporting
Held-out test set size: 2,378 samples (2,271 legitimate, 107 fraudulent).

| Metric | Calibrated Linear SVM (Champion, Thresh = 0.33) | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | 98.53% | Overall correct predictions |
| **Precision (Fraud)** | 86.00% | 86 true fraud / (86 true fraud + 14 false alarms) |
| **Recall (Fraud)** | 80.37% | 86 true fraud caught / 107 total fraud |
| **F1-Score (Fraud)** | 0.8309 | Harmonic mean of precision and recall |
| **PR-AUC** | 0.9011 | Precision-Recall curve area |
| **ROC-AUC** | 0.9912 | Receiver Operating Characteristic area |
| **Fraud-Class False Negative Rate** | **19.63%** | Miss rate: $21 / 107$ fraudulent postings missed |
| **Dataset-Wide False Negative Rate** | **0.88%** | Total missed scams across all $2,378$ test postings |
| **False Positive Rate** | **0.62%** | $14 / 2,271$ legitimate postings falsely flagged |

---

## 6. Summary of System Audit Findings

| Category | Status | Details |
| :--- | :---: | :--- |
| **Data Hygiene** | ✅ Healthy | 2,031 duplicates removed; 70/15/15 stratified split; no cross-contamination. |
| **Leakage Prevention** | ✅ Healthy | Preprocessing fits strictly on training folds; test set completely held-out. |
| **Parity** | ✅ Healthy | End-to-end scikit-learn Pipeline used consistently in API and training. |
| **Testing** | ✅ Healthy | 12 automated unit and integration tests passing in `pytest` (100% pass rate). |
| **Database** | ✅ Implemented | SQLite database (`data/jobshield.db`) storing `scan_history` and `user_feedback`. |
| **Model Diversity** | ✅ Healthy | 6 algorithms benchmarked and verified in `MODEL_COMPARISON.csv`. |
| **SVM Calibration** | ✅ Verified | `CalibratedClassifierCV` Platt scaling correctly fitted and explained. |
| **Explainability** | ✅ Healthy | Signal attribution, red flags, green flags, and responsible-use notices implemented. |
| **Browser Extension** | ⚠️ Developer Prototype | Manifest V3 extension in `extension/` runs locally unpacked against localhost API; not on Chrome Web Store. |

---

## 7. Actionable Takeaways & Architecture Boundaries

### Implemented & Verified in Code
1. Zero-leakage data processing with 3-way stratified split and duplicate template removal.
2. Calibrated Linear SVM champion model tuned to threshold 0.33, achieving 86.00% precision and 80.37% recall.
3. SQLite database persistence with tables for historical scans and user feedback.
4. FastAPI service providing single prediction, batch prediction, history retrieval, and feedback logging.
5. Interactive web frontend (`frontend/index.html`) with real-time risk gauges and scan history table.
6. 12 automated tests in `pytest` and GitHub Actions CI workflow.

### Developer Prototype
1. **Browser Extension:** Functional locally via `chrome://extensions` (Developer Mode) querying `http://127.0.0.1:8000`. Dependent on DOM selectors and local server execution.

### Future Roadmap (Planned Enhancements)
1. **Chrome Web Store Distribution:** Packaging extension and hosting public cloud API.
2. **Contextual Text Embeddings:** Adding dense sentence-transformers (`all-MiniLM-L6-v2`) to capture paraphrased scam phrases.
3. **Domain Age & WHOIS Verification:** Live domain registration checking to catch enterprise impersonation blind spots (the primary source of false negatives).
4. **API Rate Limiting:** Adding `slowapi` to prevent API misuse.
