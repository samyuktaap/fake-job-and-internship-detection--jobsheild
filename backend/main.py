"""
backend/main.py
FastAPI application for JobShield: Fraudulent Job Detection API.
Provides endpoints for inference, batch processing, user feedback,
scan history persistence, model transparency metrics, and error analysis.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import (
    JobAnalysisRequest,
    JobAnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    SignalDetail,
    FeatureContribution,
    FeedbackRequest,
    FeedbackResponse,
    ScanHistoryItem
)
from backend.database import (
    init_db,
    save_scan,
    get_recent_scans,
    get_scan_by_id,
    save_feedback,
    get_recent_feedback
)
from src.explainability.explainer import JobShieldExplainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jobshield-api")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Initialize database tables
init_db()

app = FastAPI(
    title="JobShield API",
    description="NLP-powered decision-support system to detect fraudulent job listings.",
    version="2.0.0"
)

# Enable CORS for local development and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Explainer and Model Pipeline
explainer = JobShieldExplainer()


@app.get("/api/health", summary="Health Check")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "JobShield API",
        "version": "2.0.0",
        "pipeline_loaded": explainer.pipeline is not None,
        "decision_threshold": explainer.threshold
    }


def _process_single_job(payload: JobAnalysisRequest, persist: bool = True) -> JobAnalysisResponse:
    job_dict = payload.model_dump()
    result = explainer.explain_prediction(job_dict)

    red_flags = [SignalDetail(**rf) for rf in result.get("red_flags", [])]
    green_flags = [SignalDetail(**gf) for gf in result.get("green_flags", [])]
    top_contribs = [FeatureContribution(**fc) for fc in result.get("top_feature_contributions", [])]

    scan_id = None
    if persist:
        try:
            scan_data = {
                "title": payload.title,
                "company": payload.company_profile[:80] if payload.company_profile else "",
                "location": payload.location or "",
                "risk_score_pct": result["risk_score_pct"],
                "calibrated_probability": result["calibrated_probability"],
                "prediction_label": result["prediction_label"],
                "risk_level": result["risk_level"],
                "badge_color": result["badge_color"],
                "red_flags": [rf.model_dump() for rf in red_flags],
                "green_flags": [gf.model_dump() for gf in green_flags]
            }
            scan_id = save_scan(scan_data)
        except Exception as db_err:
            logger.warning(f"Failed to record scan in database: {str(db_err)}")

    return JobAnalysisResponse(
        scan_id=scan_id,
        prediction_label=result["prediction_label"],
        risk_level=result["risk_level"],
        badge_color=result["badge_color"],
        risk_score_pct=result["risk_score_pct"],
        calibrated_probability=result["calibrated_probability"],
        decision_threshold_used=result["decision_threshold_used"],
        red_flags=red_flags,
        green_flags=green_flags,
        key_genuine_signals=result.get("key_genuine_signals", []),
        top_feature_contributions=top_contribs,
        responsible_use_disclaimer=result["responsible_use_disclaimer"]
    )


@app.post("/api/predict", response_model=JobAnalysisResponse, summary="Analyze Job Posting")
def analyze_job_posting(payload: JobAnalysisRequest) -> JobAnalysisResponse:
    """
    Accepts a single job posting text and metadata, applies the champion ML pipeline,
    records scan in SQLite history, and returns calibrated risk analysis with explanations.
    """
    try:
        return _process_single_job(payload, persist=True)
    except Exception as e:
        logger.error(f"Error analyzing job posting: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/api/predict/batch", response_model=BatchAnalysisResponse, summary="Batch Analyze Job Postings")
def batch_analyze_jobs(payload: BatchAnalysisRequest) -> BatchAnalysisResponse:
    """
    Accepts up to 50 job postings in a single request, processes each using the calibrated
    champion pipeline, and records scans in history.
    """
    try:
        results = [_process_single_job(job, persist=True) for job in payload.jobs]
        return BatchAnalysisResponse(
            total_analyzed=len(results),
            results=results
        )
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )


@app.get("/api/history", response_model=List[ScanHistoryItem], summary="Get Scan History")
def get_scan_history(limit: int = Query(default=20, ge=1, le=100)) -> List[ScanHistoryItem]:
    """
    Retrieves recent job scans recorded in SQLite persistence.
    """
    try:
        scans = get_recent_scans(limit=limit)
        items = []
        for s in scans:
            rf = [SignalDetail(**flag) for flag in s.get("red_flags", [])]
            gf = [SignalDetail(**flag) for flag in s.get("green_flags", [])]
            items.append(ScanHistoryItem(
                id=s["id"],
                created_at=s["created_at"],
                title=s["title"],
                company=s.get("company", ""),
                location=s.get("location", ""),
                risk_score_pct=s["risk_score_pct"],
                calibrated_probability=s["calibrated_probability"],
                prediction_label=s["prediction_label"],
                risk_level=s["risk_level"],
                badge_color=s["badge_color"],
                red_flags=rf,
                green_flags=gf
            ))
        return items
    except Exception as e:
        logger.error(f"Error retrieving scan history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@app.post("/api/feedback", response_model=FeedbackResponse, summary="Submit User Feedback")
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """
    Records human reviewer feedback or candidate corrections on job predictions,
    supporting human-in-the-loop retraining workflows.
    """
    try:
        feedback_id = save_feedback(payload.model_dump())
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="success",
            message="Thank you! Your feedback has been recorded for model evaluation and audit."
        )
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback persistence error: {str(e)}"
        )


@app.get("/api/feedback", summary="Get User Feedback Entries")
def get_feedback_list(limit: int = Query(default=50, ge=1, le=100)) -> List[Dict[str, Any]]:
    """
    Retrieves recent feedback submissions.
    """
    try:
        return get_recent_feedback(limit=limit)
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@app.get("/api/metrics", summary="Model Comparison & Evaluation Metrics")
def get_metrics() -> Dict[str, Any]:
    """
    Returns comparative evaluation metrics on the held-out test split
    across all models.
    """
    comp_file = os.path.join(ARTIFACTS_DIR, "model_comparison.json")
    meta_file = os.path.join(ARTIFACTS_DIR, "champion_metadata.json")
    audit_file = os.path.join(ARTIFACTS_DIR, "audit_metrics.json")

    comparison = {}
    metadata = {}
    audit_data = {}

    if os.path.exists(comp_file):
        with open(comp_file, "r") as f:
            comparison = json.load(f)
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            metadata = json.load(f)
    if os.path.exists(audit_file):
        with open(audit_file, "r") as f:
            audit_data = json.load(f)

    return {
        "models": comparison,
        "champion": metadata,
        "audit": audit_data
    }


@app.get("/api/samples", summary="Curated Job Test Cases")
def get_samples() -> List[Dict[str, Any]]:
    """
    Returns representative sample job listings (both legitimate and fraudulent)
    for instant interactive demonstration.
    """
    sample_file = os.path.join(ARTIFACTS_DIR, "sample_test_cases.json")
    if not os.path.exists(sample_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample cases not found."
        )
    with open(sample_file, "r") as f:
        return json.load(f)


@app.get("/api/error-analysis", summary="Held-out Test Error Analysis")
def get_error_analysis() -> Dict[str, Any]:
    """
    Returns inspected False Positives and False Negatives on the held-out test split.
    """
    error_file = os.path.join(ARTIFACTS_DIR, "error_analysis.json")
    if not os.path.exists(error_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Error analysis not found."
        )
    with open(error_file, "r") as f:
        return json.load(f)


# Serve frontend static assets if available
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "JobShield API is running. Frontend index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
