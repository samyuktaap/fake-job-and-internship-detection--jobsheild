# JobShield: Machine Learning Fraudulent Job Detection

[![JobShield CI](https://github.com/samyuktaap/fake-job-and-internship-detection--jobsheild/actions/workflows/ci.yml/badge.svg)](https://github.com/samyuktaap/fake-job-and-internship-detection--jobsheild/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-emerald?logo=fastapi)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-blue)

> **An NLP and supervised machine learning system to analyze job advertisements and estimate whether a listing is potentially fraudulent.**

JobShield is built in strict adherence to the **JobShield Group Work Plan & Model Development Guide**, emphasizing honest generalization, zero-leakage validation, fraud-class precision and recall balance, transparent error analysis, and clear distinction between production-ready code, local prototypes, and planned roadmap items.

---

## 📌 Implementation Status Breakdown

To ensure complete architectural clarity, JobShield's features are categorized into completed working implementations, local developer prototypes, and planned enhancements:

### ✅ Completed & Tested (Production-Ready Core)
- **Zero-Leakage Benchmark Partition:** Strictly deduplicated by job description body; 14,583 unique listings with 0% description overlap between train, val, and test splits (70/15/15 stratified split).
- **Calibrated Linear SVM Champion:** Linear SVM with Platt scaling calibration (`CalibratedClassifierCV(LinearSVC, cv=3)`) achieving optimal fraud-class precision (**83.91%**) and recall (**78.49%**, F1: **0.8111**) at validation-tuned threshold **0.36**.
- **Transparent Explainability:** Model weights and domain heuristics extracted via `JobShieldExplainer`, providing human-readable red flags, legitimacy markers, and top feature contributions.
- **Persistent SQLite Database:** Fully operational SQLite database layer (`data/jobshield.db`) storing scan history (`scan_history` table) and candidate feedback (`user_feedback` table) via `backend/database.py`.
- **FastAPI REST API:** 7 live endpoints (`/api/health`, `/api/predict`, `/api/predict/batch`, `/api/history`, `/api/feedback`, `/api/metrics`, `/api/samples`).
- **Comprehensive Automated Test Suite:** **12 automated unit and integration tests** passing in `pytest` (100% pass rate) testing preprocessing, feature extraction, explainability, SQLite persistence, batch scanning, and API validation.
- **Continuous Integration (CI):** GitHub Actions workflow (`.github/workflows/ci.yml`) automatically executing test validation and linting on every push and pull request.
- **Interactive Web Dashboard:** Vanilla HTML5/Tailwind web app (`frontend/index.html`) featuring real-time risk gauges, sample presets, interactive feedback submission, and scan history table.

### 🧪 Developer Prototype (Local Unpacked Extension)
- **Chromium Browser Extension (`extension/`):** Manifest V3 Chrome/Edge extension for local developer testing.
  - Interacts with job listings on LinkedIn and Indeed via DOM scraping (`content.js`).
  - Queries the local FastAPI backend (`http://127.0.0.1:8000/api/predict`).
  - *Current Status:* Unpacked developer prototype loaded locally via `chrome://extensions` (Developer Mode). Requires the JobShield Uvicorn server running locally. Relies on active DOM selectors for job details.

### 📋 Planned Future Roadmap (Not Yet Implemented)
- **Chrome Web Store Distribution:** Packaging, asset review, and public web store listing.
- **Production Cloud Deployment:** Public HTTPS reverse-proxy hosting for the FastAPI backend.
- **Semantic Sentence Embeddings:** Contextual transformer representations (`all-MiniLM-L6-v2`) in parallel with TF-IDF.
- **External Domain & WHOIS Verification:** Live domain age checking and recruiter email verification to catch enterprise impersonation blind spots.
- **API Rate Limiting:** Throttling endpoint usage via `slowapi`.

---

## 📊 Model Comparison on Held-Out Test Set (2,188 Postings, Zero Leakage)

All 6 algorithms were benchmarked across 5-fold cross-validation on the development set and evaluated on the untouched, zero-leakage held-out test split (2,188 postings; 93 fraudulent, 2,095 legitimate):

| Model | Feature Representation | Decision Threshold | Precision (Fraud) | Recall (Fraud) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** | Majority Baseline | 0.05 | 0.00% | 0.00% | 0.0000 | 0.0425 | 0.5000 |
| **Multinomial Naive Bayes** | TF-IDF Text | 0.33 | 63.21% | 72.04% | 0.6734 | 0.7513 | 0.9636 |
| **Logistic Regression (Balanced)** | TF-IDF + Metadata + Risk Signals | 0.70 | 75.79% | 77.42% | 0.7660 | 0.8357 | 0.9815 |
| **Linear SVM (Calibrated) ★** | TF-IDF + Metadata + Risk Signals | **0.36 (Tuned)** | **83.91%** | **78.49%** | **0.8111** | **0.8648** | **0.9815** |
| **Random Forest (Balanced)** | TF-IDF + Metadata + Risk Signals | 0.60 | 63.92% | 66.67% | 0.6526 | 0.7577 | 0.9763 |
| **XGBoost (scale_pos_weight)** | TF-IDF + Metadata + Risk Signals | 0.64 | 67.33% | 73.12% | 0.7010 | 0.8221 | 0.9715 |

### Held-Out Confusion Matrix (Linear SVM Champion, Threshold = 0.36)

```
                       Predicted Legitimate    Predicted Fraudulent     Total
Actual Legitimate (0)         2,081 (TN)                 14 (FP)        2,095
Actual Fraudulent (1)            20 (FN)                 73 (TP)           93
Total Predicted               2,101                      87            2,188
```

### Detailed Error and False Negative Breakdown

To avoid ambiguity in error reporting, false negatives are reported both at the fraud-class level and across the total dataset:

- **Fraud-Class False Negative Rate (Miss Rate):** **21.51%**  
  $$\text{FNR}_{\text{fraud}} = \frac{\text{FN}}{\text{TP} + \text{FN}} = \frac{20}{73 + 20} = \frac{20}{93} = 21.51\%$$
  The model successfully caught **78.49%** of fraudulent postings ($\text{Recall} = 73/93$) and missed 20 scam postings (~1 in 5). Qualitative analysis shows that all 20 missed scams were enterprise impersonations that plagiarized legitimate Fortune 500 job descriptions.
- **Dataset-Wide False Negative Rate:** **0.91%**  
  $$\text{FNR}_{\text{dataset}} = \frac{\text{FN}}{\text{Total Samples}} = \frac{20}{2,188} = 0.91\%$$
  Undetected scams represent only 0.91% of all job listings evaluated in the held-out test set.
- **False Positive Rate (Specificity):** **0.67%**  
  $$\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} = \frac{14}{2,081 + 14} = \frac{14}{2,095} = 0.67\%$$
  Out of 2,095 authentic job listings, only 14 were falsely flagged as suspicious ($\text{Specificity} = 99.33\%$), protecting legitimate small employers.

---

## 🛠️ Installation & Setup

### 1. Clone & Prerequisites
- Python 3.11+ (Tested on Python 3.12)
- Git

```bash
git clone https://github.com/samyuktaap/fake-job-and-internship-detection--jobsheild.git
cd fake-job-and-internship-detection--jobsheild
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests -v
```
All **12 test suites** will execute, verifying data hygiene, risk signal extraction, calibrated probability estimation, SQLite persistence, and API endpoints.

### 4. Start the JobShield Web Application & API
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
👉 **`http://127.0.0.1:8000`**

---

## 🧩 Running the Browser Extension (Developer Prototype)

The JobShield browser extension allows real-time analysis directly inside LinkedIn and Indeed job detail pages:

1. Ensure the backend server is running: `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`.
2. Open Chrome or Edge and navigate to `chrome://extensions` (or `edge://extensions`).
3. Enable **Developer mode** (toggle in top-right corner).
4. Click **Load unpacked** and select the `extension/` directory from this repository.
5. Navigate to an individual job advertisement on [LinkedIn Jobs](https://www.linkedin.com/jobs) or [Indeed](https://www.indeed.com).
6. Click the JobShield icon in your browser toolbar, then click **"Scan Current Job Listing"**.
   *(Note: The page must have the job description container loaded in view for DOM scraping).*

---

## 🔍 API Endpoints Summary

- **`GET /api/health`**: Verifies service status, model pipeline loading, and active decision threshold (0.33).
- **`POST /api/predict`**: Accepts job advertisement JSON, returns calibrated probability, risk level badge, heuristic red/green flags, top contributing features, and saves record to SQLite.
- **`POST /api/predict/batch`**: Accepts up to 50 job listings in a single payload for bulk recruitment screening.
- **`GET /api/history`**: Retrieves recent scans persisted in the SQLite database.
- **`POST /api/feedback`**: Records candidate or reviewer feedback on false positives/negatives for retraining audits.
- **`GET /api/metrics`**: Exposes cross-validation and test benchmark comparisons across all 6 models.
- **`GET /api/samples`**: Supplies 4 curated sample listings (2 genuine, 2 fraudulent) for instant interactive demonstration.
- **`GET /api/error-analysis`**: Exposes inspected False Positives and False Negatives from the held-out evaluation.

---

## ⚠️ Responsible-Use Disclaimer
JobShield is designed strictly as an **algorithmic decision-support tool**. Predictions reflect statistical associations learned from historical training data and must never be treated as legal findings or definitive accusations against employers. Job seekers are strongly advised to verify organizations independently through official business registries, confirmed company domains, and direct outreach.
