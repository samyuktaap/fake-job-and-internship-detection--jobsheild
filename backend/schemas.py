"""
backend/schemas.py
Pydantic schemas for JobShield API request and response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class JobAnalysisRequest(BaseModel):
    title: str = Field(..., description="Job advertisement title", min_length=2)
    description: str = Field(..., description="Detailed job description and responsibilities", min_length=10)
    company_profile: Optional[str] = Field(default="", description="About the company or employer")
    requirements: Optional[str] = Field(default="", description="Required qualifications, skills, or experience")
    benefits: Optional[str] = Field(default="", description="Benefits, perks, or compensation details")
    location: Optional[str] = Field(default="", description="Geographic location (e.g., US, NY, New York)")
    department: Optional[str] = Field(default="", description="Department or functional division")
    employment_type: Optional[str] = Field(default="Full-time", description="Full-time, Part-time, Contract, etc.")
    required_experience: Optional[str] = Field(default="Unspecified", description="Entry level, Mid-Senior level, etc.")
    required_education: Optional[str] = Field(default="Unspecified", description="High School, Bachelor's Degree, etc.")
    telecommuting: Optional[int] = Field(default=0, ge=0, le=1, description="1 if remote/telecommuting, else 0")
    has_company_logo: Optional[int] = Field(default=0, ge=0, le=1, description="1 if logo is present, else 0")
    has_questions: Optional[int] = Field(default=0, ge=0, le=1, description="1 if applicant screening questions exist, else 0")


class SignalDetail(BaseModel):
    category: str
    severity: Optional[str] = None
    detail: str


class FeatureContribution(BaseModel):
    feature_index: int
    weight: float
    feature_name: Optional[str] = None


class JobAnalysisResponse(BaseModel):
    scan_id: Optional[int] = None
    prediction_label: str
    risk_level: str
    badge_color: str
    risk_score_pct: float
    calibrated_probability: float
    decision_threshold_used: float
    red_flags: List[SignalDetail]
    green_flags: List[SignalDetail]
    key_genuine_signals: List[str] = []
    top_feature_contributions: List[FeatureContribution] = []
    responsible_use_disclaimer: str


class BatchAnalysisRequest(BaseModel):
    jobs: List[JobAnalysisRequest] = Field(..., min_length=1, max_length=50)


class BatchAnalysisResponse(BaseModel):
    total_analyzed: int
    results: List[JobAnalysisResponse]


class FeedbackRequest(BaseModel):
    scan_id: Optional[int] = None
    job_title: Optional[str] = ""
    reported_label: str = Field(..., description="'Legitimate', 'Fraudulent', or 'Suspicious'")
    comments: Optional[str] = ""
    user_email: Optional[str] = ""


class FeedbackResponse(BaseModel):
    feedback_id: int
    status: str
    message: str


class ScanHistoryItem(BaseModel):
    id: int
    created_at: str
    title: str
    company: Optional[str] = ""
    location: Optional[str] = ""
    risk_score_pct: float
    calibrated_probability: float
    prediction_label: str
    risk_level: str
    badge_color: str
    red_flags: List[SignalDetail] = []
    green_flags: List[SignalDetail] = []


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class MetricSet(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrix] = None


class ModelComparisonSummary(BaseModel):
    models: Dict[str, Any]
    champion_model: str
    champion_metrics: MetricSet
    optimal_threshold: float
