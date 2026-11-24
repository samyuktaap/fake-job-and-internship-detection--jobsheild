"""
src/features/preprocessor.py
Feature engineering, text preprocessing, and scikit-learn Pipeline construction.
Strictly fits on training data to prevent any data leakage.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion


SUSPICIOUS_PAYMENT_KEYWORDS = [
    r"wire\s+transfer",
    r"western\s+union",
    r"moneygram",
    r"upfront\s+fee",
    r"processing\s+fee",
    r"registration\s+fee",
    r"background\s+check\s+fee",
    r"cashier\s*'?s?\s*check",
    r"bitcoin",
    r"cryptocurrency",
    r"bank\s+account\s+details",
    r"credit\s+card\s+number",
    r"social\s+security\s+number",
    r"ssn\b",
]

FREE_EMAIL_PATTERNS = [
    r"@gmail\.com",
    r"@yahoo\.com",
    r"@hotmail\.com",
    r"@outlook\.com",
    r"@aol\.com",
    r"@yandex\.com",
    r"@protonmail\.com",
    r"@mail\.com",
]

UNREALISTIC_EARNING_PATTERNS = [
    r"earn\s+\$\d+",
    r"\$\d+\s+per\s+day",
    r"\$\d+\s+daily",
    r"make\s+thousands",
    r"unlimited\s+earning",
    r"no\s+experience\s+(?:required|needed)\s+earn",
    r"guaranteed\s+income",
    r"work\s+from\s+home\s+make\s+\$",
]

URGENCY_PATTERNS = [
    r"act\s+fast",
    r"immediate\s+start",
    r"urgent\s+hiring",
    r"limited\s+openings",
    r"positions\s+filling\s+fast",
    r"apply\s+immediately",
]


def clean_text(text: str) -> str:
    """Sanitize and clean raw text."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URL addresses
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Normalize excessive spaces and newlines
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


class JobTextExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts and joins all relevant text fields into a single unified textual document
    with clear structural section markers.
    """
    def __init__(self, include_profile: bool = True):
        self.include_profile = include_profile

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)

        texts = []
        for _, row in df.iterrows():
            title = clean_text(str(row.get("title", "")))
            profile = clean_text(str(row.get("company_profile", ""))) if self.include_profile else ""
            desc = clean_text(str(row.get("description", "")))
            reqs = clean_text(str(row.get("requirements", "")))
            benefits = clean_text(str(row.get("benefits", "")))
            location = clean_text(str(row.get("location", "")))
            department = clean_text(str(row.get("department", "")))

            # Title is repeated 2x to weight heading keywords appropriately in sparse bag-of-words
            combined = (
                f"job title: {title} {title} "
                f"location: {location} "
                f"department: {department} "
                f"company profile: {profile} "
                f"description: {desc} "
                f"requirements: {reqs} "
                f"benefits: {benefits}"
            )
            texts.append(combined.strip())
        return texts


class DomainRiskExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts tabular domain-specific engineered risk indicators from job postings.
    """
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)

        features = []
        for _, row in df.iterrows():
            full_text = " ".join([
                str(row.get("title", "")),
                str(row.get("company_profile", "")),
                str(row.get("description", "")),
                str(row.get("requirements", "")),
                str(row.get("benefits", ""))
            ]).lower()

            profile_text = str(row.get("company_profile", "")).strip()
            desc_text = str(row.get("description", "")).strip()
            reqs_text = str(row.get("requirements", "")).strip()

            # 1. Suspicious payment keywords (wire transfer, upfront fee, crypto, ssn)
            has_payment_mention = 1 if any(re.search(pat, full_text) for pat in SUSPICIOUS_PAYMENT_KEYWORDS) else 0

            # 2. Free personal email address in posting text (e.g., send cv to job@gmail.com)
            has_free_email = 1 if any(re.search(pat, full_text) for pat in FREE_EMAIL_PATTERNS) else 0

            # 3. Unrealistic earning promises
            has_unrealistic_earning = 1 if any(re.search(pat, full_text) for pat in UNREALISTIC_EARNING_PATTERNS) else 0

            # 4. Urgent hiring language
            has_urgency = 1 if any(re.search(pat, full_text) for pat in URGENCY_PATTERNS) else 0

            # 5. Missing / very short company profile (strong historical indicator in EMSCAD)
            missing_profile = 1 if len(profile_text) < 30 else 0

            # 6. Missing company logo
            has_logo = int(row.get("has_company_logo", 0))
            no_logo = 1 if has_logo == 0 else 0

            # 7. Telecommuting flag
            telecommuting = int(row.get("telecommuting", 0))

            # 8. Has screening questions
            has_questions = int(row.get("has_questions", 0))
            no_questions = 1 if has_questions == 0 else 0

            # 9. Missing requirements
            missing_reqs = 1 if len(reqs_text) < 15 else 0

            # 10. Brief or missing description
            brief_description = 1 if len(desc_text) < 100 else 0

            features.append([
                has_payment_mention,
                has_free_email,
                has_unrealistic_earning,
                has_urgency,
                missing_profile,
                no_logo,
                telecommuting,
                no_questions,
                missing_reqs,
                brief_description
            ])

        return np.array(features, dtype=float)


def get_risk_feature_names() -> List[str]:
    return [
        "risk_payment_mention",
        "risk_free_email",
        "risk_unrealistic_earning",
        "risk_urgency_language",
        "risk_missing_company_profile",
        "risk_no_company_logo",
        "risk_telecommuting_flag",
        "risk_no_screening_questions",
        "risk_missing_requirements",
        "risk_brief_description"
    ]


def build_text_only_pipeline() -> Pipeline:
    """
    Builds a text-only TF-IDF feature extraction pipeline.
    """
    return Pipeline([
        ("text_extractor", JobTextExtractor()),
        ("tfidf", TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            min_df=3,
            max_df=0.9
        ))
    ])


def build_full_feature_pipeline() -> FeatureUnion:
    """
    Builds a combined feature pipeline:
    1. TF-IDF sparse text features (up to 12k unigram/bigram tokens)
    2. One-hot encoded categorical metadata (employment_type, experience, education)
    3. Engineered domain risk signals
    """
    # 1. NLP branch
    text_branch = Pipeline([
        ("extract_text", JobTextExtractor()),
        ("tfidf", TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            min_df=3,
            max_df=0.9
        ))
    ])

    # 2. Categorical branch
    cat_cols = ["employment_type", "required_experience", "required_education"]
    cat_branch = Pipeline([
        ("selector", ColumnTransformer([
            ("cat_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)
        ]))
    ])

    # 3. Domain risk indicators branch
    risk_branch = Pipeline([
        ("risk_extractor", DomainRiskExtractor())
    ])

    combined_features = FeatureUnion([
        ("nlp", text_branch),
        ("categorical", cat_branch),
        ("domain_risk", risk_branch)
    ])

    return combined_features


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data.loader import load_raw_data, clean_and_deduplicate
    print("Testing Preprocessor pipeline on a mini batch...")
    raw = load_raw_data()
    clean_df, _ = clean_and_deduplicate(raw)
    sample = clean_df.head(20)
    
    pipe = build_full_feature_pipeline()
    X_trans = pipe.fit_transform(sample)
    print("Transformed matrix shape:", X_trans.shape)
    print("Preprocessor test passed successfully!")
