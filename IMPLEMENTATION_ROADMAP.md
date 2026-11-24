# JobShield 2.0: Implementation & Upgrade Roadmap

**Version:** 2.0.0-draft  
**Status:** Phase 1 Complete (Audit & Benchmarking Verified)  
**Target Completion:** 4-Week Sprint Schedule

---

## 🗺️ High-Level Project Phases

```
┌─────────────────────────────────┐
│ Phase 1: Audit & ML Benchmark   │  ✅ COMPLETED
│ - Architecture & Leakage Review │
│ - 6 Models 5-Fold CV & Test     │
│ - Error Analysis Case Studies   │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ Phase 2: ML Model Upgrade       │  ✅ COMPLETED
│ - Calibrated SVM Champion (0.33)│
│ - Vocabulary Signal Extraction  │
│ - Positive Legitimacy Markers   │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ Phase 3: Backend & Database     │  ✅ COMPLETED
│ - SQLite Persistence (Scans+FB) │
│ - User Feedback / Correction API│
│ - Batch Scanning API (/batch)   │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ Phase 4: Frontend Refinement    │  ✅ COMPLETED
│ - Feedback Dialog Modal         │
│ - Persistent Scan History Tab   │
│ - Live Benchmark KPIs & Matrix  │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│ Phase 5: Extension Deployment   │  ✅ COMPLETED
│ - Manifest V3 Background Worker │
│ - LinkedIn / Indeed Real-Time UI│
│ - Genuine & Risk Explanations   │
└─────────────────────────────────┘
```

---

## Detailed Sprint Specifications

### Phase 1: Project Audit & ML Upgrade (Current Phase - ✅ Complete)
- [x] Run full test suite and verify 100% pass rate.
- [x] Audit dataset quality, class imbalance (19.6 : 1), and deduplication (removed 2,031 duplicates).
- [x] Verify zero-leakage pipeline design and 70/15/15 stratified split.
- [x] Benchmark 6 classification algorithms across 5-fold CV and held-out test split.
- [x] Conduct qualitative error analysis on False Positives and False Negatives.
- [x] Generate `PROJECT_AUDIT.md`, `ML_EVALUATION_REPORT.md`, and `MODEL_COMPARISON.csv`.

---

### Phase 2: Machine Learning Enhancements & Pipeline Hardening (Sprint Week 1)
- **Goal:** Address enterprise impersonation blind spots (False Negatives) and reduce false accusations on small employers (False Positives).
- **Key Tasks:**
  1. **Deploy Calibrated Linear SVM as Primary Production Pipeline:**
     - Replace Logistic Regression champion with Calibrated Linear SVM ($F_1 = 0.8309$, Precision = $86.00\%$, PR-AUC = $0.9011$).
  2. **Incorporate Semantic Sentence Embeddings:**
     - Integrate a lightweight dense text encoder (e.g., `sentence-transformers/all-MiniLM-L6-v2`) in parallel with TF-IDF unigrams/bigrams to capture semantic intent and paraphrased scam phrases.
  3. **External Domain & Entity Signals:**
     - Extract domain names from contact emails and website URLs.
     - Add heuristic signals for domain age (newly registered domains $< 30$ days) and public email services used for enterprise hiring.
  4. **Ensemble Stacking:**
     - Build a soft-voting ensemble combining Calibrated Linear SVM + XGBoost + Dense Embeddings.

---

### Phase 3: Backend & Database Hardening (Sprint Week 2)
- **Goal:** Transform the stateless API into an audited, persistent service with user feedback loops.
- **Key Tasks:**
  1. **SQLite / SQLAlchemy Database Layer:**
     - Create tables for:
       - `scan_history`: `id`, `timestamp`, `job_title_hash`, `risk_score`, `verdict`, `red_flags_json`.
       - `user_feedback`: `scan_id`, `user_reported_label` (0 or 1), `comments`, `reviewed_status`.
  2. **Feedback & Flag Correction API:**
     - Implement `POST /api/feedback` to allow users to flag false positives and false negatives for future retraining pools.
  3. **Batch Scanning API:**
     - Implement `POST /api/predict/batch` accepting up to 50 listings per request for recruiter or job board moderation.
  4. **Rate Limiting & Security:**
     - Integrate `slowapi` to prevent API denial-of-service (e.g., 60 requests/minute per IP).

---

### Phase 4: Frontend UI/UX Refinements (Sprint Week 3)
- **Goal:** Surface new transparency controls and scan history to the user without breaking existing workflows.
- **Key Tasks:**
  1. **Interactive Feedback Dialog:**
     - Add a *"Report False Prediction"* button on the result card that triggers a non-intrusive feedback modal.
  2. **Scan History Drawer:**
     - Add a *"Recent Scans"* sidebar showing the user's past 10 inspections in local storage.
  3. **Sensitivity / Threshold Slider:**
     - Allow power users to toggle between *"Strict Protection"* (Recall-prioritized, threshold 0.50) and *"Conservative Screening"* (Precision-prioritized, threshold 0.70).
  4. **Batch CSV Scanner Tab:**
     - Allow users to drag-and-drop a CSV file of job listings and download scored risk reports.

---

### Phase 5: Browser Extension Packaging & Live Testing (Sprint Week 4)
- **Goal:** Ensure seamless, live real-time detection on LinkedIn, Indeed, and other hiring portals.
- **Key Tasks:**
  1. **Background Service Worker Integration:**
     - Finalize `extension/background.js` to route API calls and eliminate HTTPS mixed-content blocks.
  2. **Live DOM Selector Testing:**
     - Test and harden DOM extraction against real LinkedIn SPA job cards and Indeed dynamic search views.
  3. **Inline UI Badge:**
     - Inject an inline verified pill badge directly beside the job title on LinkedIn job postings.
  4. **Packaging & Distribution:**
     - Bundle unpacked extension zip file with one-click installation documentation for Chrome and Edge.
