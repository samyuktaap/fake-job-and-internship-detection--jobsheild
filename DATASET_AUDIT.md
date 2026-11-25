# JobShield: Comprehensive Dataset & Data Leakage Audit

**Role:** Senior Machine Learning Engineer  
**Date:** November 25, 2026  
**Dataset:** Employment Scam Aegean Dataset (EMSCAD)  
**File Location:** `data/fake_job_postings.csv` (splits in `data/train.csv`, `data/val.csv`, `data/test.csv`)

---

## 1. Executive Summary

This audit performs an empirical, statistical, and architectural inspection of the datasets used to train and evaluate the **JobShield** fraudulent job detection models. 

### Key Dataset Findings:
1. **Severe Class Imbalance:** The raw dataset comprises **17,880 postings**, of which only **866 (4.84%)** are fraudulent (imbalance ratio of $19.6 : 1$). After deduplication, the dataset contains **15,849 postings** with **710 (4.48%)** fraudulent samples ($21.3 : 1$ imbalance).
2. **Missing Value Asymmetry:** Legitimate postings routinely provide company profiles (81.5%), logos (79.6%), and detailed requirements (84.9%). Conversely, missing company profiles and missing logos are heavily correlated with fraud, creating high-leverage tabular heuristics that also risk penalizing legitimate early-stage startups.
3. **Template Deduplication:** Removing 2,031 3-field duplicate templates (`title`, `company_profile`, `description`) eliminated exact copy-paste syndication across cities.
4. **Critical Data Leakage Discovery (Near-Duplicate Descriptions):** An empirical cross-split audit revealed that **154 distinct job description bodies overlap between `train.csv` and `test.csv`**, spanning 376 training rows and 202 test rows (including 16 fraudulent test postings).
   - On the overlapping test subset, the champion model achieved an artificial **100% recall (16/16 caught)**.
   - On the strictly disjoint test subset ($N=2,176$), recall dropped to **76.92% (70/91 caught)**.
   - This proves that bag-of-words memorization of identical description paragraphs artificially inflated overall test recall from **76.92%** to **80.37%**.
5. **Temporal & Semantic Drift:** EMSCAD was compiled in 2012–2014 from Workable postings. Modern 2026 recruitment scams heavily utilize Telegram/WhatsApp interviews, crypto tasks, and LLM-synthesized job descriptions that are absent from EMSCAD.

---

## 2. Raw Dataset Profiling (`data/fake_job_postings.csv`)

- **Total Rows:** 17,880
- **Total Columns:** 18
- **Target Variable:** `fraudulent` (Binary: `0` = Legitimate, `1` = Fraudulent)

### 2.1 Column-by-Column Missingness and Utility Audit

| Column Name | Data Type | Missing Count | Missing % | Primary Semantic Role in Modeling |
| :--- | :---: | :---: | :---: | :--- |
| `job_id` | Integer | 0 | 0.00% | Unique primary key (excluded from feature matrix). |
| `title` | String | 0 | 0.00% | High-signal text (weighted 2x in `JobTextExtractor`). |
| `location` | String | 346 | 1.94% | Country/State/City string (e.g., `US, NY, New York`). |
| `department` | String | 11,547 | 64.58% | Sparse text; concatenated into full text representation. |
| `salary_range` | String | 15,012 | 83.96% | Extremely sparse (e.g., `100000-120000`); unutilized in dense ML. |
| `company_profile` | String | 3,308 | 18.50% | **Critical feature:** Absence strongly correlates with scam postings. |
| `description` | String | 1 | 0.01% | Core body text; primary source of TF-IDF n-grams. |
| `requirements` | String | 2,696 | 15.08% | Technical requirements and qualifications text. |
| `benefits` | String | 7,212 | 40.34% | Perks, health, vacation, 401(k) text. |
| `telecommuting` | Binary | 0 | 0.00% | Remote indicator (1 if remote, 0 otherwise). |
| `has_company_logo` | Binary | 0 | 0.00% | **Critical indicator:** 1 if corporate logo attached, 0 otherwise. |
| `has_questions` | Binary | 0 | 0.00% | 1 if applicant screening questionnaire included. |
| `employment_type` | Categorical | 3,471 | 19.41% | Full-time, Part-time, Contract, Temporary, Other. |
| `required_experience`| Categorical | 7,050 | 39.43% | Entry level, Mid-Senior, Associate, Director, Executive. |
| `required_education` | Categorical | 8,105 | 45.33% | Bachelor's Degree, High School, Master's, Unspecified. |
| `industry` | Categorical | 4,903 | 27.42% | High cardinality categorical (e.g., IT, Healthcare). |
| `function` | Categorical | 6,455 | 36.10% | High cardinality functional role (e.g., Engineering, Sales). |
| `fraudulent` | Binary | 0 | 0.00% | Ground-truth supervision label. |

---

## 3. Class Imbalance Analysis

Job fraud is an inherently extreme-imbalance classification problem:

### Raw Class Distribution:
- **Legitimate (`0`):** 17,014 (95.16%)
- **Fraudulent (`1`):** 866 (4.84%)
- **Imbalance Ratio:** $19.64 : 1$

### Deduplicated Class Distribution (15,849 Samples):
- **Legitimate (`0`):** 15,139 (95.52%)
- **Fraudulent (`1`):** 710 (4.48%)
- **Imbalance Ratio:** $21.32 : 1$

### Machine Learning Implications:
- A trivial majority classifier that blindly predicts "legitimate" for every job achieves an deceptive **95.52% accuracy** while catching **0% of scams**.
- Consequently, accuracy is an invalid selection criterion. System design must focus strictly on **Fraud-Class Recall**, **Fraud-Class Precision**, **Fraud-Class F1-Score**, and **Precision-Recall AUC (PR-AUC)**.
- Models trained under standard empirical risk minimization without loss reweighting fail to predict the minority class. The current architecture successfully addresses this via `class_weight="balanced"` in Linear SVM and Logistic Regression, and `scale_pos_weight=21.3` in XGBoost.

---

## 4. Deduplication & Data Leakage Audit

### 4.1 Stage 1: Exact and Template Deduplication
In online recruitment portals, scammers frequently syndicate the identical posting across hundreds of geographic areas to maximize victim reach. A standard random cross-validation or train/test split inadvertently places copies of identical postings into both training and test sets, causing artificial $99\%+$ test accuracy that collapses in production.

- **Exact Duplicate Rows:** 0 (all rows possessed unique `job_id` values).
- **Text Template Duplicates:** Removed **2,031 rows** that shared identical `('title', 'company_profile', 'description')`.
- **Cleaned Dataset Total:** 15,849 unique listings.

### 4.2 Stage 2: Empirical Cross-Split Leakage Audit (Discovery)
To verify whether deduplication on the 3-field tuple was sufficient, an exhaustive intersection audit was executed across the description strings in `data/train.csv`, `data/val.csv`, and `data/test.csv`.

#### Cross-Split Description Overlaps:
- `train.csv` $\cap$ `val.csv`: **169** overlapping description strings.
- `train.csv` $\cap$ `test.csv`: **154** overlapping description strings.
- `val.csv` $\cap$ `test.csv`: **65** overlapping description strings.

#### Detailed Inspection of the 154 Train-Test Overlapping Descriptions:
- **Rows in `train.csv` affected:** 376 rows (including 28 fraud cases).
- **Rows in `test.csv` affected:** 202 rows (including 16 fraud cases).

#### Why Did This Leakage Occur?
1. **Recruiter Boilerplate:** Legitimate recruitment agencies posted multiple positions with varying titles (e.g., *"Avionics Systems Design and Analysis Engineer"* vs. *"Junior Avionics Systems Design and Analysis Engineer"*) while copying the exact same job description text.
2. **Scam syndication with altered titles:** Scammers syndicated the same phishing description under slightly different titles (e.g., *"Administrative Assistant"* vs. *"Data Entry Clerk"*).

#### Empirical Proof of Performance Inflation:
Evaluating the champion Calibrated Linear SVM ($0.33$ threshold) separately on the overlapping and non-overlapping test partitions reveals the exact impact of this leakage:

```
┌────────────────────────────────────────────────────────────────────────┐
│             Test Split Performance Under Leakage Audit                 │
├──────────────────────────┬───────────┬───────────┬──────────┬──────────┤
│ Partition                │ Samples   │ Fraud (N) │ Recall   │ F1-Score │
├──────────────────────────┼───────────┼───────────┼──────────┼──────────┤
│ Overlapping Subset       │ 202       │ 16        │ 100.00%  │ 1.0000   │
│ Strictly Disjoint Subset │ 2,176     │ 91        │ 76.92%   │ 0.8000   │
├──────────────────────────┼───────────┼───────────┼──────────┼──────────┤
│ Aggregate Test Set       │ 2,378     │ 107       │ 80.37%   │ 0.8309   │
└──────────────────────────┴───────────┴───────────┴──────────┴──────────┘
```

**Senior ML Engineer Finding:** The model achieved a perfect **100% recall (16/16)** on the overlapping scam listings because its TF-IDF n-grams had already memorized those exact paragraphs during training. On truly novel, disjoint job listings, the true zero-shot fraud recall is **76.92%**.

---

## 5. Train / Validation / Test Split Integrity

The dataset is partitioned into three splits using stratified sampling:

| Split Name | Sample Count | Legitimate (`0`) | Fraudulent (`1`) | Fraud Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set (`train.csv`)** | 11,093 (70.0%) | 10,596 | 497 | 4.48% |
| **Validation Set (`val.csv`)** | 2,378 (15.0%) | 2,272 | 106 | 4.46% |
| **Held-Out Test Set (`test.csv`)** | 2,378 (15.0%) | 2,271 | 107 | 4.50% |
| **Total** | **15,849 (100.0%)** | **15,139** | **710** | **4.48%** |

### Split Protocol Verification:
- **Stratification:** Class distributions across all three splits are strictly preserved at $4.48\% \pm 0.02\%$.
- **Preprocessing Isolation:** Vectorizers, scalers, and imputers in `src/models/train.py` are fitted **strictly on `train.csv`**. Neither `val.csv` nor `test.csv` was used to calculate IDF frequencies or category vocabularies.
- **Threshold Freezing:** Optimal threshold ($0.33$) was selected exclusively on `val.csv` before running the single held-out test evaluation on `test.csv`.

---

## 6. Real-World Generalization & Out-of-Distribution Vulnerabilities

An essential responsibility of an ML audit is to evaluate how well models will perform on real job postings in production (e.g., LinkedIn, Indeed in 2026):

### 6.1 Vulnerability 1: Enterprise Impersonation (Zero-Shot Text Plagiarism)
- **Mechanism:** A scammer copies the exact text and branding of an authentic enterprise listing (e.g., Google, Amazon, Deloitte) and redirects applicants to a spoofed form or Telegram interview.
- **Audit Result:** Because the text contains authentic corporate vocabulary and describes genuine responsibilities, bag-of-words and lexical models assign a scam probability near zero ($0.0076$). All 21 test False Negatives exhibited this pattern.
- **Required Fix:** Text-based NLP must be paired with **domain verification** (verifying applicant submission URLs and recruiter email domains against authentic business records).

### 6.2 Vulnerability 2: Modern 2026 Scam Modalities
- **Mechanism:** Scammers today rarely ask for "Western Union" or "cashier's checks" directly in the job description. Instead, they advertise remote social media, rating, or data entry positions with minimal initial requirements, deferring the scam (crypto deposits, paid training kits, identity theft) to post-application communication.
- **Audit Result:** EMSCAD contains only historical 2012–2014 listings. Without dense semantic embeddings that understand modern conversational scam intent, static TF-IDF fails.

### 6.3 Vulnerability 3: Startups and Small Business Penalization (False Positives)
- **Mechanism:** Legitimate early-stage startups often lack official logos, extensive corporate profiles, or formal screening questions.
- **Audit Result:** The 10 heuristic risk features assign heavy penalties to listings lacking logos or corporate profiles, generating 14 false positive flags in the test set.

---

## 7. Recommendations for Dataset Upgrades (JobShield 2.0)

1. **Implement Content-Hash Deduplication on Description Bodies:**
   - Prior to partitioning data for JobShield 2.0, deduplicate strictly on normalized `description` text to guarantee 100% disjoint splits without leakage.
2. **Incorporate Modern 2026 Scam Data:**
   - Synthesize and collect modern recruitment scam vectors (Telegram interviews, task-based crypto schemes, phishing Google Forms) to update the training distribution.
3. **Active Learning Feedback Integration:**
   - Leverage the existing `user_feedback` table in SQLite (`data/jobshield.db`) to log real-world user corrections and build quarterly retraining candidate datasets.
