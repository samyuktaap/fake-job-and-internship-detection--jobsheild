"""
tests/test_api.py
Integration tests for FastAPI endpoints: single prediction, batch prediction,
scan persistence history, user feedback, transparency metrics, and error states.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["pipeline_loaded"] is True
    assert 0.1 <= data["decision_threshold"] <= 0.9


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "champion" in data
    assert "audit" in data
    assert data["champion"]["champion_model"] == "Linear SVM (Calibrated)"
    assert 0.1 <= data["champion"]["selected_threshold"] <= 0.9
    assert data["champion"]["test_metrics"]["accuracy"] > 0.95


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_predict_endpoint_valid_and_persisted():
    payload = {
        "title": "Senior Cloud Infrastructure Engineer",
        "description": "Design and automate AWS/GCP cloud environments with Terraform, Kubernetes, and Prometheus.",
        "company_profile": "Enterprise cloud services firm providing reliable mission-critical deployments.",
        "requirements": "5+ years DevOps or SRE experience with Go, Python, and CI/CD pipelines.",
        "benefits": "Competitive salary, 401k match, health insurance, paid parental leave.",
        "location": "US, CA, San Francisco",
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "required_education": "Bachelor's Degree",
        "has_company_logo": 1,
        "has_questions": 1,
        "telecommuting": 0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_label"] in ["Likely Genuine", "Likely Genuine (Review Recommended)", "Potentially Fraudulent"]
    assert "risk_score_pct" in data
    assert "responsible_use_disclaimer" in data
    assert "scan_id" in data
    assert data["scan_id"] is not None
    assert isinstance(data["scan_id"], int)


def test_predict_endpoint_fraud_case():
    payload = {
        "title": "URGENT Data Entry Clerk / Cashier Check Processing",
        "description": "Make $800 daily! We will send you a cashier check. Wire transfer back money via Western Union. Contact hr@gmail.com immediately.",
        "company_profile": "",
        "requirements": "None",
        "benefits": "Unlimited cash",
        "location": "US, NY",
        "employment_type": "Part-time",
        "required_experience": "Entry level",
        "required_education": "High School",
        "has_company_logo": 0,
        "has_questions": 0,
        "telecommuting": 1
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Fraudulent" in data["prediction_label"]
    assert data["risk_score_pct"] > 50.0
    assert len(data["red_flags"]) > 0


def test_predict_endpoint_invalid_short_description():
    payload = {
        "title": "Job",
        "description": "Too short"  # min length is 10
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_endpoint():
    batch_payload = {
        "jobs": [
            {
                "title": "DevOps Engineer",
                "description": "Maintain CI/CD pipelines, manage Kubernetes clusters on AWS with Terraform.",
                "company_profile": "Standard enterprise software company.",
                "requirements": "Docker, Kubernetes, Linux.",
                "has_company_logo": 1
            },
            {
                "title": "Work from Home Money Transfer Agent",
                "description": "Receive company wire transfers and forward cashier checks via Western Union.",
                "company_profile": "",
                "requirements": "Bank account",
                "has_company_logo": 0
            }
        ]
    }
    response = client.post("/api/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_analyzed"] == 2
    assert len(data["results"]) == 2
    assert "risk_score_pct" in data["results"][0]
    assert "risk_score_pct" in data["results"][1]


def test_history_endpoint():
    response = client.get("/api/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    first_item = data[0]
    assert "id" in first_item
    assert "title" in first_item
    assert "risk_score_pct" in first_item
    assert "prediction_label" in first_item


def test_feedback_endpoint():
    # Submit feedback on a scan
    feedback_payload = {
        "scan_id": 1,
        "job_title": "Senior Cloud Infrastructure Engineer",
        "reported_label": "Legitimate",
        "comments": "Verified company domain and official interview invitation received.",
        "user_email": "auditor@example.com"
    }
    response = client.post("/api/feedback", json=feedback_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "feedback_id" in data
    assert isinstance(data["feedback_id"], int)

    # Retrieve feedback list
    get_res = client.get("/api/feedback?limit=10")
    assert get_res.status_code == 200
    fb_list = get_res.json()
    assert isinstance(fb_list, list)
    assert len(fb_list) >= 1
