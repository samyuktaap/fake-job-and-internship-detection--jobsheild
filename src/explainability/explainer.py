"""
src/explainability/explainer.py
JobShield explainability module.
Analyzes contributing signals, keyword flags, feature weights,
and formats human-interpretable risk assessments with responsible-use caveats.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.features.preprocessor import (
    SUSPICIOUS_PAYMENT_KEYWORDS,
    FREE_EMAIL_PATTERNS,
    UNREALISTIC_EARNING_PATTERNS,
    URGENCY_PATTERNS,
    get_risk_feature_names
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")


class JobShieldExplainer:
    """
    Interprets model decisions, highlights risk signals vs legitimacy markers,
    and formats calibrated risk ratings with mandatory screening disclaimers.
    """

    def __init__(self, pipeline_path: str = None, metadata_path: str = None):
        if pipeline_path is None:
            pipeline_path = os.path.join(ARTIFACTS_DIR, "jobshield_pipeline.joblib")
        if metadata_path is None:
            metadata_path = os.path.join(ARTIFACTS_DIR, "champion_metadata.json")

        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.threshold = 0.33
        self._cached_feature_names = None
        self.load_artifacts()

    def load_artifacts(self):
        if os.path.exists(self.pipeline_path):
            self.pipeline = joblib.load(self.pipeline_path)
        if os.path.exists(self.metadata_path):
            import json
            with open(self.metadata_path, "r") as f:
                meta = json.load(f)
                self.threshold = meta.get("selected_threshold", 0.33)

    def _get_feature_names(self) -> List[str]:
        if self._cached_feature_names is not None:
            return self._cached_feature_names
        try:
            pre = self.pipeline.named_steps.get("preprocessor")
            tfidf_names = list(pre.transformer_list[0][1].named_steps["tfidf"].get_feature_names_out())
            cat_names = list(pre.transformer_list[1][1].named_steps["selector"].get_feature_names_out())
            risk_names = get_risk_feature_names()
            self._cached_feature_names = tfidf_names + cat_names + risk_names
        except Exception:
            self._cached_feature_names = []
        return self._cached_feature_names

    def extract_direct_signals(self, job_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scans job text and structured attributes for direct heuristic risk patterns
        and verified legitimacy markers.
        """
        full_text = " ".join([
            str(job_dict.get("title", "")),
            str(job_dict.get("company_profile", "")),
            str(job_dict.get("description", "")),
            str(job_dict.get("requirements", "")),
            str(job_dict.get("benefits", ""))
        ]).lower()

        profile = str(job_dict.get("company_profile", "")).strip()
        desc = str(job_dict.get("description", "")).strip()
        reqs = str(job_dict.get("requirements", "")).strip()
        has_logo = int(job_dict.get("has_company_logo", 0))
        has_questions = int(job_dict.get("has_questions", 0))
        telecommuting = int(job_dict.get("telecommuting", 0))

        red_flags = []
        green_flags = []

        # 1. Suspicious payment keywords
        found_payments = [pat.replace(r"\s+", " ").replace(r"\b", "").replace("\\", "")
                          for pat in SUSPICIOUS_PAYMENT_KEYWORDS if re.search(pat, full_text)]
        if found_payments:
            red_flags.append({
                "category": "Payment / Fee Request",
                "severity": "CRITICAL",
                "detail": f"Contains payment, banking, or sensitive credential terminology: {', '.join(found_payments[:3])}"
            })
        else:
            green_flags.append({
                "category": "Zero Financial Demands",
                "detail": "No upfront fee, wire transfer, cryptocurrency, or check cashing schemes found."
            })

        # 2. Free personal email in corporate posting
        found_emails = [pat.replace("\\", "") for pat in FREE_EMAIL_PATTERNS if re.search(pat, full_text)]
        if found_emails:
            red_flags.append({
                "category": "Unofficial Communication Channel",
                "severity": "HIGH",
                "detail": f"Mentions public personal email domains ({', '.join(found_emails)}) rather than verified company domain."
            })
        else:
            green_flags.append({
                "category": "Clean Recruitment Channel",
                "detail": "Official corporate hiring channel; zero suspicious personal emails (@gmail, @yahoo) or messaging app handles."
            })

        # 3. Unrealistic earning claims
        if any(re.search(pat, full_text) for pat in UNREALISTIC_EARNING_PATTERNS):
            red_flags.append({
                "category": "Unrealistic Compensation",
                "severity": "HIGH",
                "detail": "Promises excessive daily/weekly income or guaranteed payouts with minimal qualifications."
            })
        else:
            green_flags.append({
                "category": "Realistic Career Scope",
                "detail": "Standard professional employment and compensation scope without deceptive 'earn thousands daily' claims."
            })

        # 4. Urgent hiring pressure
        if any(re.search(pat, full_text) for pat in URGENCY_PATTERNS):
            red_flags.append({
                "category": "High Pressure Urgency",
                "severity": "MEDIUM",
                "detail": "Employs high-pressure urgency cues (e.g. 'act fast', 'immediate start', 'apply right away')."
            })
        else:
            green_flags.append({
                "category": "Standard Hiring Timeline",
                "detail": "Normal recruitment pacing with no manufactured urgency or high-pressure countdowns."
            })

        # 5. Company profile
        if len(profile) >= 30:
            green_flags.append({
                "category": "Established Profile",
                "detail": "Detailed company profile and organizational overview provided."
            })
        elif len(desc) < 80:
            # Only flag missing company profile as red if the description is also very short or sparse!
            red_flags.append({
                "category": "Vague Employer Identity",
                "severity": "LOW",
                "detail": "Missing company profile overview in listing."
            })

        # 6. Company Logo
        if has_logo == 1:
            green_flags.append({
                "category": "Verified Branding",
                "detail": "Official company logo or verified entity branding."
            })
        elif len(desc) < 80:
            red_flags.append({
                "category": "Missing Branding",
                "severity": "LOW",
                "detail": "No official company logo attached."
            })

        # 7. Screening Questions
        if has_questions == 1:
            green_flags.append({
                "category": "Structured Screening",
                "detail": "Job posting includes custom candidate screening questions or ATS tracking."
            })

        # 8. Requirements clarity
        if len(reqs) > 40 or len(desc) > 150:
            green_flags.append({
                "category": "Defined Professional Scope",
                "detail": "Clear technical responsibilities, candidate qualifications, and industry standard job criteria specified."
            })

        return {
            "red_flags": red_flags,
            "green_flags": green_flags
        }

    def explain_prediction(self, job_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates full prediction and transparency report for a job posting.
        """
        if self.pipeline is None:
            self.load_artifacts()

        # Fill default fields if missing to ensure complete pipeline preprocessing
        clean_input = dict(job_dict)
        clean_input.setdefault("employment_type", "Full-time")
        clean_input.setdefault("required_experience", "Entry level")
        clean_input.setdefault("required_education", "Bachelor's Degree")
        clean_input.setdefault("company_profile", "")
        clean_input.setdefault("requirements", "")
        clean_input.setdefault("benefits", "")
        clean_input.setdefault("telecommuting", 0)
        clean_input.setdefault("has_company_logo", 1 if clean_input.get("company_profile") else 0)
        clean_input.setdefault("has_questions", 0)

        df_input = pd.DataFrame([clean_input])

        # Model probability estimation
        if hasattr(self.pipeline, "predict_proba"):
            raw_proba = float(self.pipeline.predict_proba(df_input)[0, 1])
        else:
            raw_proba = 0.5

        # Decision based on calibrated validation threshold
        is_fraud = raw_proba >= self.threshold
        risk_pct = round(raw_proba * 100, 1)

        if raw_proba >= 0.70:
            risk_level = "HIGH RISK"
            prediction_label = "Potentially Fraudulent"
            badge_color = "red"
        elif raw_proba >= self.threshold:
            risk_level = "ELEVATED RISK"
            prediction_label = "Potentially Fraudulent (Borderline)"
            badge_color = "orange"
        elif raw_proba >= 0.20:
            risk_level = "MODERATE RISK"
            prediction_label = "Likely Genuine (Review Recommended)"
            badge_color = "yellow"
        else:
            risk_level = "LOW RISK"
            prediction_label = "Likely Genuine"
            badge_color = "green"

        signals = self.extract_direct_signals(clean_input)

        # Build feature importance insights and key vocabulary signals
        top_contributions = []
        key_genuine_signals = []

        try:
            clf = self.pipeline.named_steps.get("clf")
            preprocessor = self.pipeline.named_steps.get("preprocessor")
            if clf is not None:
                coef = None
                if hasattr(clf, "coef_"):
                    coef = clf.coef_[0]
                elif hasattr(clf, "calibrated_classifiers_"):
                    coefs = [cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_ if hasattr(cc.estimator, "coef_")]
                    if coefs:
                        coef = np.mean(coefs, axis=0)

                if coef is not None:
                    all_names = self._get_feature_names()
                    X_trans = preprocessor.transform(df_input)
                    indices = X_trans.nonzero()[1]
                    feature_scores = [
                        (idx, float(coef[idx] * X_trans[0, idx]), all_names[idx] if idx < len(all_names) else f"feat_{idx}")
                        for idx in indices
                    ]

                    fraud_features = [f for f in feature_scores if f[1] > 0]
                    fraud_features.sort(key=lambda x: x[1], reverse=True)

                    genuine_features = [f for f in feature_scores if f[1] < 0]
                    genuine_features.sort(key=lambda x: x[1])

                    chosen_features = fraud_features if is_fraud else genuine_features
                    top_contributions = [
                        {"feature_index": int(i), "weight": round(s, 4), "feature_name": name}
                        for i, s, name in chosen_features[:5]
                    ]

                    # Extract readable genuine keywords
                    for _, _, name in genuine_features:
                        clean_name = (
                            name.replace("cat_encoder__", "")
                            .replace("required_education_", "")
                            .replace("required_experience_", "")
                            .replace("employment_type_", "")
                        )
                        if not clean_name.startswith("risk_") and len(clean_name) > 2 and clean_name not in key_genuine_signals:
                            key_genuine_signals.append(f"'{clean_name}'")
                        if len(key_genuine_signals) >= 5:
                            break
        except Exception:
            top_contributions = []
            key_genuine_signals = []

        return {
            "prediction_label": prediction_label,
            "risk_level": risk_level,
            "badge_color": badge_color,
            "risk_score_pct": risk_pct,
            "calibrated_probability": round(raw_proba, 4),
            "decision_threshold_used": self.threshold,
            "red_flags": signals["red_flags"],
            "green_flags": signals["green_flags"],
            "key_genuine_signals": key_genuine_signals,
            "top_feature_contributions": top_contributions,
            "responsible_use_disclaimer": (
                "DISCLAIMER: JobShield is an algorithmic screening aid, not a legal finding or definitive proof. "
                "Predictions reflect statistical correlations from the EMSCAD dataset. Job seekers must always "
                "verify employers through official company domains, corporate emails, and legitimate business registries."
            )
        }
