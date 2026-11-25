# JobShield 2.0: Implementation & Upgrade Roadmap

**Version:** 2.0.0  
**Status:** Audit & Code Alignment Verified  
**Schedule:** Sprint Architecture Overview

---

## 🗺️ High-Level Project Status

```
┌───────────────────────────────────────┐
│ Phase 1: Audit & ML Benchmark         │  ✅ COMPLETED
│ - Architecture & Leakage Review       │
│ - 6 Models 5-Fold CV & Held-Out Test  │
│ - Error Analysis Case Studies         │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│ Phase 2: ML Model Upgrade             │  ✅ COMPLETED (Core) / 📋 ROADMAP (Dense)
│ - Calibrated SVM Champion (0.33) [x]  │
│ - Signal Extraction & Markers [x]     │
│ - Dense Transformer Embeddings [ ]    │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│ Phase 3: Backend & Database           │  ✅ COMPLETED (Core) / 📋 ROADMAP (Rate Limit)
│ - SQLite Persistence (Scans+FB) [x]   │
│ - User Feedback / Correction API [x]  │
│ - Batch Scanning API (/batch) [x]     │
│ - Rate Limiting (slowapi) [ ]         │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│ Phase 4: Frontend Refinement          │  ✅ COMPLETED
│ - Feedback Dialog Modal [x]           │
│ - Persistent Scan History Tab [x]     │
│ - Live Benchmark KPIs & Matrix [x]    │
└───────────────────┬───────────────────┘
                    │
┌───────────────────▼───────────────────┐
│ Phase 5: Extension Deployment         │  🧪 PROTOTYPE (Local) / 📋 ROADMAP (Store)
│ - Manifest V3 Background Worker [x]   │
│ - LinkedIn / Indeed Local Scraper [x] │
│ - Chrome Web Store Deployment [ ]     │
└───────────────────────────────────────┘
```

---

## Detailed Sprint Specifications & Feature Verification

### Phase 1: Project Audit & ML Benchmark (✅ Completed & Verified)
- [x] Run full automated test suite (12/12 passing).
- [x] Audit dataset quality, class imbalance (19.6 : 1), and deduplication (removed 2,031 duplicates).
- [x] Verify zero-leakage pipeline design and 70/15/15 stratified split.
- [x] Benchmark 6 classification algorithms across 5-fold CV and held-out test split.
- [x] Conduct qualitative error analysis on False Positives and False Negatives.
- [x] Generate `PROJECT_AUDIT.md`, `ML_EVALUATION_REPORT.md`, and `MODEL_COMPARISON.csv`.

---

### Phase 2: Machine Learning Enhancements & Pipeline Hardening
- **Completed Features:**
  - [x] **Calibrated Linear SVM Champion:** Deployed `CalibratedClassifierCV(LinearSVC, cv=3)` with Platt scaling.
  - [x] **Threshold Tuning:** Selected optimal cutoff **0.33** on validation data to achieve $86.00\%$ precision and $80.37\%$ recall.
  - [x] **Feature Attribution & Heuristics:** Implemented `JobShieldExplainer` extracting averaged fold weights and 10 domain risk signals.
- **Future Roadmap (Planned Enhancements):**
  - [ ] **Semantic Sentence Embeddings:** Integrate `sentence-transformers/all-MiniLM-L6-v2` in parallel with TF-IDF.
  - [ ] **External Domain WHOIS Verification:** API lookup for domain age (<30 days) to prevent enterprise impersonation.
  - [ ] **Ensemble Soft-Voting:** Stacking Calibrated Linear SVM + XGBoost + Dense Embeddings.

---

### Phase 3: Backend & Database Hardening
- **Completed Features:**
  - [x] **SQLite Database Layer (`backend/database.py`):**
    - `scan_history` table: persists `title`, `company`, `risk_score_pct`, `calibrated_probability`, `red_flags`, `green_flags`.
    - `user_feedback` table: persists user-reported corrections and review comments.
  - [x] **Feedback & Correction API:** Implemented `POST /api/feedback`.
  - [x] **Batch Scanning API:** Implemented `POST /api/predict/batch` accepting up to 50 listings per request.
  - [x] **Scan History API:** Implemented `GET /api/history` returning persisted scans.
- **Future Roadmap (Planned Enhancements):**
  - [ ] **Rate Limiting:** Integrate `slowapi` to protect endpoints against automated scraping.
  - [ ] **API Authentication:** Add optional API keys for recruiter / enterprise batch consumers.

---

### Phase 4: Frontend UI/UX Refinements
- **Completed Features:**
  - [x] **Interactive Feedback Dialog:** Modal dialog allowing users to flag incorrect predictions directly to SQLite.
  - [x] **Persistent Scan History View:** Dynamic history table pulling historical scans from `/api/history`.
  - [x] **KPI Cards & Confusion Matrix:** Visual display of model metrics on held-out test set.
  - [x] **Responsible-Use Disclaimers:** Visible algorithmic screening notices on all prediction outputs.

---

### Phase 5: Browser Extension Deployment
- **Completed Developer Prototype (`extension/`):**
  - [x] **Manifest V3 Core:** Service worker (`background.js`), popup UI (`popup.html`, `popup.js`), and content script (`content.js`).
  - [x] **Local Job Scraper:** Extracts job title, company, description, and metadata from LinkedIn and Indeed pages.
  - [x] **Local Inference Integration:** Sends payload to local JobShield API (`http://127.0.0.1:8000/api/predict`) and displays risk score.
  - [x] **Manual Unpacked Loading:** Successfully loaded and tested in Chrome/Edge via `chrome://extensions` ("Load unpacked").
- **Future Roadmap (Planned Enhancements):**
  - [ ] **Chrome Web Store Submission:** Publish packaged extension to Chrome Web Store.
  - [ ] **Cloud Backend Integration:** Point extension to a hosted HTTPS API instead of localhost.
  - [ ] **MutationObserver Scrapers:** Auto-detect SPA navigation transitions on LinkedIn and Indeed search feeds.
