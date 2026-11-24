# JobShield 2.0: Comprehensive Project & Architecture Audit

**Date:** November 24, 2026  
**Auditor:** JobShield Engineering & ML Team  
**Scope:** Complete codebase inspection covering backend, frontend, preprocessing, ML models, dataset hygiene, data leakage, and test infrastructure.

---

## 1. Executive Summary

This audit establishes the baseline status of the **JobShield** project. The existing application provides a functional, leakage-free machine learning pipeline and a FastAPI service with an interactive browser frontend. The system focuses on decision-support screening for fraudulent job advertisements using NLP and tabular metadata from the Employment Scam Aegean Dataset (EMSCAD).

All 8 existing automated unit and integration tests pass. The training and inference pipelines maintain strict preprocessing parity. No data leakage was found between splits. However, gaps exist in persistent storage (no database), advanced feature representations (embeddings/contextual encoders), API rate-limiting, and automated feedback collection.

---

## 2. Codebase & System Architecture Review

### 2.1 Backend Architecture & API Routes (`backend/`)
- **Framework:** FastAPI (`0.141.1`) served via Uvicorn on port `8000`.
- **API Endpoints Implemented:**
  - `GET /api/health`: Health status, model pipeline load state, active decision threshold.
  - `POST /api/predict`: Validates payload via Pydantic `JobAnalysisRequest`, executes `JobShieldExplainer.explain_prediction()`, and returns calibrated risk score, verdict badge, detected warning signals, and legitimacy markers.
  - `GET /api/metrics`: Returns comparative evaluation metrics across models and champion metadata.
  - `GET /api/samples`: Delivers 4 curated demo job listings (2 legitimate, 2 fraudulent).
  - `GET /api/error-analysis`: Exposes inspected False Positives and False Negatives from the held-out test set.
  - `GET /`: Serves `frontend/index.html` static single-page application.
- **Middleware:** Wildcard CORS (`allow_origins=["*"]`) enabled for local web and extension access.
- **Validation:** Pydantic v2 schemas (`backend/schemas.py`) enforce input length requirements (e.g., description >= 10 chars).

### 2.2 Frontend Architecture (`frontend/`)
- **Technology:** Vanilla HTML5, JavaScript (ES6+), and Tailwind CSS (CDN). No heavy npm build step required.
- **Components:**
  - **Analyzer Tab:** Interactive job submission form with quick demo presets, real-time risk gauge (0–100%), detected red/green flags list, and a mandatory responsible-use disclaimer.
  - **Model Benchmark Tab:** Comparative model evaluation table, top KPI cards, and held-out test confusion matrix visualizer.
  - **Error Analysis Tab:** Detailed breakdown of False Positives and False Negatives with root-cause explanations.
  - **Methodology Tab:** 5-member project role allocation and EMSCAD dataset card.

### 2.3 Database Structure
- **Current Status: None (Stateless In-Memory).**
- Predictions, submissions, and metrics are handled entirely in-memory and via serialized filesystem artifacts (`artifacts/`).
- User-submitted job postings are discarded after inference (preserving applicant privacy).
- *Gap:* No audit log table, no persistent record of historical scans, and no user feedback storage for model retraining.

### 2.4 Testing Setup (`tests/`)
- **Framework:** `pytest` (v9.1.1) with `fastapi.testclient`.
- **Test Suites:**
  - `tests/test_pipeline.py`: Tests HTML sanitization (`clean_text`), domain risk indicator extraction (`DomainRiskExtractor`), and pipeline inference (`JobShieldExplainer`).
  - `tests/test_api.py`: Tests `/api/health`, `/api/metrics`, `/api/samples`, valid `/api/predict` payloads, and 422 validation errors on malformed payloads.
- **Test Status:** **8 passed in 1.82 seconds (100% pass rate).**

---

## 3. Data Pipeline & Quality Audit

### 3.1 Raw Dataset Profile (`data/fake_job_postings.csv`)
- **Dataset:** Employment Scam Aegean Dataset (EMSCAD).
- **Total Initial Rows:** 17,880
- **Total Columns:** 18 (e.g., `title`, `company_profile`, `description`, `requirements`, `benefits`, `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `fraudulent`).
- **Raw Class Distribution:**
  - Legitimate (`0`): 17,014 (95.16%)
  - Fraudulent (`1`): 866 (4.84%)
  - Imbalance Ratio: ~19.6 : 1

### 3.2 Missing Value Profiling
- `salary_range`: 15,012 missing (84.0%) — unreliable for dense modeling.
- `department`: 11,547 missing (64.6%).
- `required_education`: 8,105 missing (45.3%).
- `required_experience`: 7,050 missing (39.4%).
- `company_profile`: 3,308 missing (18.5%) — *crucial signal: absence strongly correlates with fraudulent listings.*
- `requirements`: 2,696 missing (15.1%).
- `description`: Only 1 row missing (handled with empty string conversion).

### 3.3 Deduplication & Leakage Prevention Audit
In recruitment datasets, scammers frequently copy-paste identical scam postings across dozens of cities. If split randomly, the exact same text appears in both training and test sets, causing artificial 99%+ accuracy that fails completely in production.
- **Exact Row Duplicates:** 282 rows removed.
- **Text Template Duplicates:** 1,749 rows removed (postings sharing identical `title`, `company_profile`, and `description`).
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
- Both training (`src/models/train.py`) and live inference (`src/explainability/explainer.py` via `backend/main.py`) load the exact same serialized pipeline artifact (`artifacts/jobshield_pipeline.joblib`).
- The explainer passes the raw input dictionary through the exact same fitted transformers. **Parity is 100% verified.**

---

## 5. Summary of System Audit Findings

| Category | Status | Details |
| :--- | :---: | :--- |
| **Data Hygiene** | ✅ Healthy | 2,031 duplicates removed; 70/15/15 stratified split; no cross-contamination. |
| **Leakage Prevention** | ✅ Healthy | Preprocessing fits strictly on training folds; test set completely held-out. |
| **Parity** | ✅ Healthy | End-to-end scikit-learn Pipeline used consistently in API and training. |
| **Testing** | ✅ Healthy | 8 automated tests passing in 1.82s. |
| **Database** | ⚠️ Incomplete | No database exists; requests are purely stateless. |
| **Model Diversity** | ✅ Healthy | 6 algorithms benchmarked (Dummy, Naive Bayes, Logistic Regression, Linear SVM, Random Forest, XGBoost). |
| **Threshold Tuning** | ✅ Healthy | Validation-tuned threshold of 0.70 slashes false positives from 96 to 25. |
| **Explainability** | ✅ Healthy | Signal attribution, red flags, green flags, and responsible-use notices implemented. |
| **Browser Extension** | ⚠️ Prototype | Extension files created in `extension/`, but not yet integrated or deployed to live browser. |

---

## 6. Actionable Takeaways: What Works, Incomplete, and Needs Fixing

### What Already Works
1. Robust, leak-free data ingestion, cleaning, and 3-way stratified splitting.
2. Production-grade scikit-learn pipeline combining 12,000 TF-IDF features with domain risk flags.
3. Calibrated Linear SVM and Logistic Regression achieving 0.78–0.83 F1 and 0.83–0.90 PR-AUC on held-out test data.
4. Fast, functional FastAPI backend with Pydantic validation and comprehensive test coverage.
5. Interactive web dashboard with demo presets, real-time risk gauges, and confusion matrix displays.

### What Is Incomplete
1. **Persistent Database:** No storage for submitted jobs, scan history, or accuracy audit logs.
2. **User Feedback Loop:** No mechanism for end users or recruiters to flag false positives or report emerging scam patterns.
3. **Batch Prediction:** Backend currently only accepts one posting at a time.
4. **Browser Extension:** Manifest V3 extension code exists in `extension/` but requires packaging and real-world testing.

### What Needs Fixing
1. **False Positive Penalties for Startups:** Legitimate early-stage companies and small businesses without logos or formal corporate profiles are penalized by heuristic risk flags.
2. **Enterprise Impersonation Blind Spots:** Scammers who copy authentic Fortune 500 job descriptions word-for-word bypass text and keyword filters.
3. **API Hardening:** Absence of rate-limiting (`slowapi`) and API key authentication.

### What Should Be Preserved
1. The **zero-leakage pipeline design** (never fit vectorizers on the full dataset).
2. The **validation threshold tuning strategy** (tuning threshold on validation data to control false positives on imbalanced data).
3. The **mandatory responsible-use disclaimer** (reinforcing that JobShield is a screening decision-support aid, not a legal finding).

### What Needs To Be Added (JobShield 2.0)
1. **SQLite / PostgreSQL Persistence:** Store scan logs, anonymized text hashes, and user feedback.
2. **Contextual Text Embeddings:** Upgrade from pure TF-IDF to transformer-based embeddings (e.g., Sentence-Transformers / MiniLM) or ensemble voting to catch paraphrased scams.
3. **Live URL & Domain Verification:** Check employer domains against WHOIS creation dates and Google Safe Browsing APIs to catch enterprise impersonation.
4. **Active Learning Feedback Loop:** Allow users to submit false positive/negative corrections for quarterly model retraining.
