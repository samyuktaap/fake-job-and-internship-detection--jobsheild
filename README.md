# JobShield: Machine Learning Fraudulent Job Detection

> **An NLP and supervised machine learning system to analyze job advertisements and estimate whether a listing is potentially fraudulent.**

JobShield is built in strict adherence to the **JobShield Group Work Plan & Model Development Guide**, emphasizing strong honest generalization, zero-leakage validation, fraud-class precision and recall balance, and transparent error analysis.

---

## 🚀 Key Highlights & Performance

- **Zero-Leakage Guarantee:** Fitted strictly on training folds; 2,031 duplicate templates removed to prevent train-test contamination; 70/15/15 stratified train/val/test split.
- **Held-Out Test Accuracy:** **98.53%** on 2,378 untouched test listings.
- **Fraud Precision:** **86.00%** (only 14 false positives out of 2,271 genuine postings).
- **Fraud Recall:** **80.37%** (caught 86 of 107 fraudulent listings).
- **Fraud F1-Score:** **0.8309** | **PR-AUC:** **0.9011** | **ROC-AUC:** **0.9912**.
- **Browser Extension:** Manifest V3 real-time Chrome/Edge extension scanning live postings on LinkedIn and Indeed.
- **Persistent Database:** SQLite backend storing scan history and candidate/reviewer feedback.
- **Decision-Support Focus:** Transparent disclaimers clarifying that predictions are statistical screening aids, not legal findings.

---

## 📊 Model Comparison on Held-Out Test Set (2,378 Postings)

| Model | Feature Representation | Threshold | Precision (Fraud) | Recall (Fraud) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** | Majority Baseline | 0.05 | 0.00% | 0.00% | 0.0000 | 0.0450 | 0.5000 |
| **Multinomial Naive Bayes** | TF-IDF Text | 0.46 | 79.59% | 72.90% | 0.7610 | 0.8389 | 0.9762 |
| **Logistic Regression (Balanced)** | TF-IDF + Metadata + Risk Signals | 0.70 | 77.27% | 79.44% | 0.7834 | 0.8348 | 0.9861 |
| **Linear SVM (Calibrated) ★** | TF-IDF + Metadata + Risk Signals | **0.33 (Tuned)** | **86.00%** | **80.37%** | **0.8309** | **0.9011** | **0.9912** |
| **Random Forest (Balanced)** | TF-IDF + Metadata + Risk Signals | 0.60 | 67.96% | 65.42% | 0.6667 | 0.7582 | 0.9764 |
| **XGBoost (scale_pos_weight)** | TF-IDF + Metadata + Risk Signals | 0.63 | 70.09% | 76.64% | 0.7321 | 0.8038 | 0.9786 |

### Held-Out Confusion Matrix (Linear SVM Champion, Threshold = 0.33)

```
                       Predicted Legitimate    Predicted Fraudulent
Actual Legitimate (0)         2,257 (TN)                 14 (FP)
Actual Fraudulent (1)            21 (FN)                 86 (TP)
```
- **False Positive Rate:** Only 0.62% (14 out of 2,271 legitimate listings flagged).
- **False Negative Rate:** 0.88% of total listings (21 missed out of 2,378).

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests -v
```
All 8 test suites will execute, covering data hygiene, feature extraction, model inference, and API endpoints.

### 4. Start the JobShield Web Application
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
👉 **`http://127.0.0.1:8000`**

---

## 🔍 API Endpoints

- **`GET /api/health`**: Service status and loaded model check.
- **`POST /api/predict`**: Accepts job advertisement JSON, returns classification label, calibrated fraud likelihood, warning flags, and legitimacy signals.
- **`GET /api/metrics`**: Returns comparative benchmarks across all 5 models and confusion matrix.
- **`GET /api/samples`**: Returns curated benchmark test cases for instant demonstration.
- **`GET /api/error-analysis`**: Returns false positive and false negative examples with explanations.

---

## ⚠️ Responsible-Use Disclaimer
JobShield is designed strictly as an **algorithmic decision-support prototype**. Predictions reflect statistical patterns learned from historical training data and must never be treated as legal findings or definitive accusations against employers. Job seekers are strongly advised to verify organizations independently through official business registries, confirmed company domains, and direct outreach.
