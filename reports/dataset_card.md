# Dataset Card: Employment Scam Aegean Dataset (EMSCAD)

## 1. Summary & Provenance
- **Dataset Name:** Employment Scam Aegean Dataset (EMSCAD / Kaggle Real or Fake Job Postings)
- **Domain:** Online job recruitment and employment scam detection
- **Dataset Provenance:** Collected by University of the Aegean researchers (Vidros et al., 2017) from authentic online recruitment portals over 2012–2014.
- **License:** Open Access / Research Use (CC0 / Open Data Commons)

## 2. Dataset Composition
- **Total Initial Records:** 17,880 listings
- **Target Feature:** `fraudulent` (Binary classification: `0` = Legitimate, `1` = Fraudulent)
- **Raw Class Distribution:**
  - Legitimate (`0`): 17,014 (95.16%)
  - Fraudulent (`1`): 866 (4.84%)
  - Imbalance Ratio: ~19.6 : 1

## 3. Data Hygiene & Leakage Prevention Strategy
### Deduplication Protocol
In the raw dataset, scammers often repost identical job descriptions across multiple locations. If left unaddressed, identical or near-identical listings would leak across training and test splits, artificially inflating reported test accuracy to near 100%.
- **Exact Row Duplicates:** 282 rows removed.
- **Text Template Duplicates:** 1,749 rows removed (postings sharing identical `title`, `company_profile`, and `description`).
- **Cleaned Dataset Total:** 15,849 unique listings (15,139 legitimate, 710 fraudulent; 4.48% fraud prevalence).

### Stratified Splitting
- **Training Set (70%):** 11,093 samples (497 fraud, 10,596 legit)
- **Validation Set (15%):** 2,378 samples (106 fraud, 2,272 legit)
- **Held-Out Test Set (15%):** 2,378 samples (107 fraud, 2,271 legit)
- **Leakage Prevention:** TF-IDF vectorizers, categorical encoders, and risk extractors are fitted **strictly on the training split**. The validation split is used solely for model selection and threshold tuning; the held-out test split is evaluated exactly once after all parameters are frozen.

## 4. Key Fields and Data Types
| Field | Type | Description | Handling |
| :--- | :--- | :--- | :--- |
| `title` | Text | Job headline / title | Cleaned, lowercased, TF-IDF extracted (weighted 2x) |
| `description` | Text | Primary job duties | HTML tags stripped, URL sanitized, TF-IDF extracted |
| `company_profile` | Text | Description of the hiring organization | TF-IDF extracted; presence and character length extracted as risk signal |
| `requirements` | Text | Candidate qualifications | TF-IDF extracted; presence checked |
| `benefits` | Text | Compensation, perks, insurance | TF-IDF extracted |
| `has_company_logo` | Binary | 1 if official logo attached, else 0 | Categorical indicator |
| `has_questions` | Binary | 1 if applicant questions exist, else 0 | Categorical indicator |
| `telecommuting` | Binary | 1 if remote position, else 0 | Categorical indicator |
| `employment_type` | Categorical | Full-time, Part-time, Contract, etc. | One-hot encoded |
| `required_experience` | Categorical | Entry level, Mid-Senior, Executive | One-hot encoded |
| `required_education` | Categorical | High School, Bachelor's, Master's | One-hot encoded |

## 5. Limitations & Ethical Considerations
1. **Temporal Decay:** Dataset dates back to 2012–2014. Modern LLM-assisted phishing schemes use more sophisticated vocabulary than legacy copy-paste templates.
2. **Language Bias:** The overwhelming majority of listings are in English. Performance will degrade on non-English postings.
3. **Screening Support Only:** The output of this classifier is a risk indicator and decision-support aid. It should never be treated as legal proof or a definitive accusation of fraud.
