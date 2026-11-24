# JobShield 2.0: Machine Learning Evaluation & Benchmarking Report

**Date:** November 24, 2026  
**Evaluation Scope:** 5-Fold Stratified Cross-Validation & Untouched Held-Out Test Evaluation  
**Dataset:** EMSCAD (15,849 unique, deduplicated postings; 4.48% fraud prevalence)

---

## 1. Evaluation Protocol & Methodology

Evaluating fraud detection requires extreme methodological discipline. Because fraudulent job postings represent only **4.48%** of the dataset, standard accuracy is dangerously deceptive. A trivial baseline predicting every posting as "legitimate" achieves 95.5% accuracy while catching zero scams.

To guarantee honest generalization and avoid data leakage:
1. **Deduplication:** 2,031 exact and template duplicates were removed before splitting, preventing identical job templates from leaking across folds.
2. **5-Fold Stratified Cross-Validation:** Conducted across the development pool (13,471 postings; 603 fraud, 12,868 legitimate) to measure model variance and stability.
3. **Validation Decision Threshold Optimization:** Candidate decision thresholds $[0.05, 0.90]$ were scanned on validation folds to maximize fraud F1-score while maintaining a minimum recall floor of 70%.
4. **Untouched Held-Out Test Set:** A pristine 15% test split (2,378 postings; 107 fraud, 2,271 legitimate) was evaluated exactly once after all models and thresholds were frozen.

---

## 2. Comprehensive Model Benchmarking Table

The table below summarizes the 5-fold cross-validation performance alongside the final held-out test results across all 6 evaluated architectures:

| Model | Feature Representation | 5-Fold CV F1 (Mean ± Std) | 5-Fold CV PR-AUC | Optimal Thresh | Test Acc | Test Prec (Fraud) | Test Rec (Fraud) | Test F1 (Fraud) | Test PR-AUC | Test ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** | None (Majority Baseline) | $0.0000 \pm 0.0000$ | 0.0447 | 0.05 | 95.50% | 0.00% | 0.00% | 0.0000 | 0.0450 | 0.5000 |
| **Multinomial Naive Bayes** | TF-IDF Text (Unigrams/Bigrams) | $0.6994 \pm 0.0477$ | 0.7765 | 0.46 | 97.94% | 79.59% | 72.90% | 0.7610 | 0.8389 | 0.9762 |
| **Logistic Regression (Balanced)** | TF-IDF + Metadata + Risk Signals | $0.6487 \pm 0.0139$ | 0.8173 | **0.70** | **98.02%** | **77.27%** | **79.44%** | **0.7834** | **0.8348** | **0.9861** |
| **Linear SVM (Calibrated) ★** | TF-IDF + Metadata + Risk Signals | **$0.8145 \pm 0.0270$** | **0.8829** | **0.33** | **98.53%** | **86.00%** | **80.37%** | **0.8309** | **0.9011** | **0.9912** |
| **Random Forest (Balanced)** | TF-IDF + Metadata + Risk Signals | $0.6076 \pm 0.0076$ | 0.7690 | 0.60 | 97.06% | 67.96% | 65.42% | 0.6667 | 0.7582 | 0.9764 |
| **XGBoost (scale_pos_weight)** | TF-IDF + Metadata + Risk Signals | $0.6940 \pm 0.0180$ | 0.8291 | 0.63 | 97.48% | 70.09% | 76.64% | 0.7321 | 0.8038 | 0.9786 |

*★ **Linear SVM (Calibrated)** achieved the highest cross-validation and test scores (F1: 0.8309, PR-AUC: 0.9011). **Logistic Regression (Balanced)** provides comparable performance (F1: 0.7834) with direct feature coefficient interpretability.*

---

## 3. Confusion Matrix Breakdown (Held-Out Test Set: 2,378 Samples)

### 3.1 Linear SVM (Calibrated, Threshold = 0.33)
```
                             Predicted Legitimate    Predicted Fraudulent     Total True
Actual Legitimate (0)              2,257 (TN)                 14 (FP)            2,271
Actual Fraudulent (1)                 21 (FN)                 86 (TP)              107
Total Predicted                    2,278                      100                2,378
```
- **False Positive Rate:** Only **0.62%** (14 out of 2,271 legitimate listings falsely flagged).
- **Fraud Recall:** Catches **80.37%** (86 of 107 fraudulent listings caught).

### 3.2 Logistic Regression (Balanced, Threshold = 0.70)
```
                             Predicted Legitimate    Predicted Fraudulent     Total True
Actual Legitimate (0)              2,246 (TN)                 25 (FP)            2,271
Actual Fraudulent (1)                 22 (FN)                 85 (TP)              107
Total Predicted                    2,268                      110                2,378
```
- **False Positive Rate:** Only **1.10%** (25 out of 2,271 legitimate listings falsely flagged).
- **Fraud Recall:** Catches **79.44%** (85 of 107 fraudulent listings caught).

### 3.3 XGBoost (scale_pos_weight, Threshold = 0.63)
```
                             Predicted Legitimate    Predicted Fraudulent     Total True
Actual Legitimate (0)              2,236 (TN)                 35 (FP)            2,271
Actual Fraudulent (1)                 25 (FN)                 82 (TP)              107
Total Predicted                    2,261                      117                2,378
```

---

## 4. Decision Threshold Optimization Rationale

In class-imbalanced fraud detection, the default classification threshold ($0.50$) is rarely optimal:

1. **Default Threshold Problem:**
   - Under `class_weight="balanced"`, the model penalizes fraud classification errors proportionally to class imbalance ($21.3 : 1$).
   - At threshold $0.50$, Logistic Regression produced **96 False Positives** on the validation set (Precision of only 50.0%). For every 1 scam identified, 1 legitimate employer was falsely accused.
2. **Validation Tuning:**
   - Scanning thresholds from $0.05$ to $0.90$ on validation data revealed that increasing the threshold to **0.70** maintained a high recall floor ($81.13\%$ on validation, $79.44\%$ on test) while cutting false positives from 96 down to 25 (a **74% reduction in false accusations**).
3. **Calibrated SVM Tuning:**
   - Because Platt scaling maps SVM decision margins to calibrated probabilities, the optimal validation threshold for Calibrated Linear SVM was **0.33**, yielding the highest precision ($86.00\%$) and highest F1 ($0.8309$) on the test set.

---

## 5. In-Depth Error Analysis

An audited, responsible ML system must examine exactly where and why it fails. Below are concrete case studies from the held-out test set:

### 5.1 False Positive Case Studies (Legitimate Listings Flagged as Fraud)

#### Case FP-1: Job #3319 — *"Looking for supplemental income & flex. hours? VOA Retail Openings!"*
- **True Label:** Legitimate (0) | **Model Probability:** 0.7944 (Predicted Fraud)
- **Attributes:** `company_profile`: Empty (`NaN`), `has_company_logo`: 0, `telecommuting`: 0
- **Root Cause:** Volunteers of America (VOA) posted a short retail thrift-store opening with no company profile and no logo. The headline contained trigger phrases like *"supplemental income"* and *"flex hours"*, which the model heavily associates with work-from-home scams.

#### Case FP-2: Job #11538 — *"Human Resources Administrator"*
- **True Label:** Legitimate (0) | **Model Probability:** 0.9764 (Predicted Fraud)
- **Attributes:** `company_profile`: Empty (`NaN`), `has_company_logo`: 0, `telecommuting`: 1
- **Root Cause:** A legitimate small agency hiring a remote HR admin. The combination of **missing company profile + missing logo + remote flag** triggered almost all non-text tabular risk indicators, overshadowing standard HR vocabulary.

#### Case FP-3: Job #6690 — *"Online Project Consultant"*
- **True Label:** Legitimate (0) | **Model Probability:** 0.8520 (Predicted Fraud)
- **Attributes:** `company_profile`: Empty (`NaN`), `has_company_logo`: 0, `description`: *"It is an ONLINE job - you work from home..."*
- **Root Cause:** Uses identical syntax to classic marketing affiliate schemes ("All backgrounds welcome", "work from home").

### 5.2 False Negative Case Studies (Scam Postings Missed by Model)

#### Case FN-1: Job #5871 — *"Receptionist Office Asst."* (Urban Asset Holdings)
- **True Label:** Fraudulent (1) | **Model Probability:** 0.2572 (Predicted Legitimate)
- **Attributes:** `company_profile`: Fully detailed corporate text (300+ chars), `has_company_logo`: 1, `has_questions`: 1
- **Root Cause:** **Enterprise Impersonation / Profile Plagiarism.** The scammers copy-pasted authentic real-estate portfolio descriptions and attached an authentic logo. The text contains zero scam keywords (no wire transfer, no upfront fee). The scam demand was deferred to private email communication after application.

#### Case FN-2: Job #2751 — *"Administrative Assistant"* (AZCO INC.)
- **True Label:** Fraudulent (1) | **Model Probability:** 0.3420 (Predicted Legitimate)
- **Attributes:** `company_profile`: Full industrial engineering profile, `has_company_logo`: 1
- **Root Cause:** Authentic text copied from an employee-owned construction firm. Bag-of-words and linear models cannot verify whether the poster actually represents AZCO INC. without external domain verification.

#### Case FN-3: Job #2928 — *"LEAN Program Manager, Carol Stream, IL"*
- **True Label:** Fraudulent (1) | **Model Probability:** 0.4694 (Predicted Legitimate)
- **Attributes:** Rich corporate jargon ("strategic enterprise-level initiatives", "Master Black Belt certification required").
- **Root Cause:** Highly sophisticated corporate identity theft that completely avoided colloquial scam vocabulary.

---

## 6. Key Conclusions & Recommendations for JobShield 2.0

1. **Linear SVM & Logistic Regression Outperform Complex Tree Models on Sparse Text:**
   - Linear SVM (Calibrated) and Logistic Regression achieved superior test F1 ($0.83$ and $0.78$) compared to Random Forest ($0.66$) and XGBoost ($0.73$). High-dimensional sparse TF-IDF spaces are fundamentally linear-separable.
2. **Text Alone Is Insufficient to Prevent Enterprise Impersonation:**
   - 100% of the investigated False Negatives were scams that plagiarized authentic enterprise descriptions.
   - **Recommendation:** Implement **external domain/email verification** (verifying that the recruiter's email domain matches the official corporate domain, and checking domain age via WHOIS).
3. **Contextual Encodings:**
   - Replace or augment n-gram TF-IDF with a lightweight transformer (e.g., `all-MiniLM-L6-v2`) to capture syntactic intent rather than static keyword presence.
