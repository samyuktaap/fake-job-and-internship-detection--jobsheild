"""
tests/test_pipeline.py
Unit tests for data processing, feature pipelines, model inference, and explainability.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.features.preprocessor import (
    clean_text,
    JobTextExtractor,
    DomainRiskExtractor,
    build_full_feature_pipeline,
    SUSPICIOUS_PAYMENT_KEYWORDS
)
from src.explainability.explainer import JobShieldExplainer


def test_clean_text():
    raw_html = "<p>Join our <b>dynamic team</b>! Visit https://example.com today.   Lots   of spaces.</p>"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "lots of spaces." in cleaned


def test_risk_extractor():
    extractor = DomainRiskExtractor()
    sample_df = pd.DataFrame([{
        "title": "Data Entry",
        "company_profile": "",
        "description": "Urgent hiring! Please wire transfer $500 processing fee and email your ssn to test@gmail.com",
        "requirements": "",
        "benefits": "",
        "has_company_logo": 0,
        "telecommuting": 1,
        "has_questions": 0
    }])
    features = extractor.fit_transform(sample_df)
    assert features.shape == (1, 10)
    # Payment mention should be 1
    assert features[0, 0] == 1.0
    # Free email should be 1
    assert features[0, 1] == 1.0
    # Missing profile should be 1
    assert features[0, 4] == 1.0
    # No logo should be 1
    assert features[0, 5] == 1.0


def test_explainer_prediction():
    explainer = JobShieldExplainer()
    assert explainer.pipeline is not None

    sample_job = {
        "title": "URGENT Administrative Assistant Needed",
        "company_profile": "",
        "description": "Make $500 daily! Wire transfer money via Western Union. Send info to manager@yahoo.com",
        "requirements": "None",
        "benefits": "Fast cash",
        "location": "US, NY",
        "department": "Admin",
        "employment_type": "Part-time",
        "required_experience": "Entry level",
        "required_education": "High School",
        "telecommuting": 1,
        "has_company_logo": 0,
        "has_questions": 0
    }

    result = explainer.explain_prediction(sample_job)
    assert "prediction_label" in result
    assert "risk_score_pct" in result
    assert "red_flags" in result
    assert "responsible_use_disclaimer" in result
    assert len(result["red_flags"]) > 0
    assert result["risk_score_pct"] > 50.0
