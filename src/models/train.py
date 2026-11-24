"""
src/models/train.py
Model training, baseline comparisons, validation threshold tuning,
held-out testing, and artifact persistence.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.features.preprocessor import build_text_only_pipeline, build_full_feature_pipeline
from src.models.evaluate import calculate_metrics, find_optimal_threshold, perform_error_analysis

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train_and_evaluate_all():
    print("Loading data splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    y_train = train_df["fraudulent"].values
    y_val = val_df["fraudulent"].values
    y_test = test_df["fraudulent"].values

    print(f"Data ready: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Define Candidate Models
    models = {
        "DummyClassifier (Majority)": {
            "pipeline": Pipeline([
                ("preprocessor", build_text_only_pipeline()),
                ("clf", DummyClassifier(strategy="most_frequent"))
            ]),
            "type": "baseline",
            "features": "Baseline only"
        },
        "Multinomial Naive Bayes": {
            "pipeline": Pipeline([
                ("preprocessor", build_text_only_pipeline()),
                ("clf", MultinomialNB(alpha=0.1))
            ]),
            "type": "nlp_baseline",
            "features": "TF-IDF text"
        },
        "Logistic Regression (Balanced)": {
            "pipeline": Pipeline([
                ("preprocessor", build_full_feature_pipeline()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="liblinear", random_state=42))
            ]),
            "type": "candidate",
            "features": "TF-IDF + Categorical + Risk signals"
        },
        "Linear SVM (Calibrated)": {
            "pipeline": Pipeline([
                ("preprocessor", build_full_feature_pipeline()),
                ("clf", CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced", max_iter=2000, random_state=42, dual="auto"),
                    cv=3
                ))
            ]),
            "type": "candidate",
            "features": "TF-IDF + Categorical + Risk signals"
        },
        "Random Forest (Balanced)": {
            "pipeline": Pipeline([
                ("preprocessor", build_full_feature_pipeline()),
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1))
            ]),
            "type": "candidate",
            "features": "TF-IDF + Categorical + Risk signals"
        }
    }

    comparison_results = {}
    fitted_pipelines = {}
    val_predictions = {}

    print("\n--- Training and Evaluating Models on Validation Split ---")
    for name, info in models.items():
        print(f"\nTraining [{name}]...")
        pipe = info["pipeline"]
        pipe.fit(train_df, y_train)
        fitted_pipelines[name] = pipe

        # Validation predictions
        if hasattr(pipe, "predict_proba"):
            val_proba = pipe.predict_proba(val_df)[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)
        else:
            val_pred = pipe.predict(val_df)
            val_proba = None

        val_metrics = calculate_metrics(y_val, val_pred, val_proba)
        val_predictions[name] = {"pred": val_pred, "proba": val_proba}

        print(f"[{name}] Val Metrics at 0.5:")
        print(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f} | PR-AUC: {val_metrics.get('pr_auc')}")

        comparison_results[name] = {
            "features": info["features"],
            "type": info["type"],
            "validation_metrics_default_threshold": val_metrics
        }

    # Model Selection & Threshold Tuning on Validation Split
    # Focus on top candidate: Linear SVM (Calibrated) (audited champion)
    candidate_name = "Linear SVM (Calibrated)"
    print(f"\nTuning decision threshold for [{candidate_name}] on Validation set...")
    val_proba_cand = val_predictions[candidate_name]["proba"]
    best_thresh, tuned_val_metrics = find_optimal_threshold(y_val, val_proba_cand, min_recall=0.75)
    print(f"Optimal Decision Threshold selected: {best_thresh}")
    print(f"Tuned Val Metrics (thresh={best_thresh}):")
    print(f"  Precision: {tuned_val_metrics['precision']:.4f} | Recall: {tuned_val_metrics['recall']:.4f} | F1: {tuned_val_metrics['f1']:.4f}")

    comparison_results[candidate_name]["optimal_threshold"] = best_thresh
    comparison_results[candidate_name]["validation_metrics_tuned_threshold"] = tuned_val_metrics

    # Select Champion Model
    champion_name = candidate_name
    champion_pipeline = fitted_pipelines[champion_name]

    print("\n--- Held-Out Test Set Final Evaluation ---")
    # Evaluate all models on held-out test set
    for name, pipe in fitted_pipelines.items():
        if hasattr(pipe, "predict_proba"):
            test_proba = pipe.predict_proba(test_df)[:, 1]
            thresh = best_thresh if name == champion_name else 0.5
            test_pred = (test_proba >= thresh).astype(int)
        else:
            test_pred = pipe.predict(test_df)
            test_proba = None
            thresh = 0.5

        test_metrics = calculate_metrics(y_test, test_pred, test_proba)
        comparison_results[name]["test_metrics"] = test_metrics
        comparison_results[name]["test_threshold"] = thresh
        print(f"[{name}] (thresh={thresh}) Held-out Test Metrics:")
        print(f"  Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | PR-AUC: {test_metrics.get('pr_auc')}")

    # Perform Error Analysis on held-out test set for Champion Model
    champ_test_proba = champion_pipeline.predict_proba(test_df)[:, 1]
    champ_test_pred = (champ_test_proba >= best_thresh).astype(int)
    error_analysis = perform_error_analysis(test_df, y_test, champ_test_pred, champ_test_proba, max_examples=5)

    # Save artifacts
    print("\nSaving artifacts...")
    # 1. Joblib saved pipeline
    pipeline_path = os.path.join(ARTIFACTS_DIR, "jobshield_pipeline.joblib")
    joblib.dump(champion_pipeline, pipeline_path)
    print(f"Champion pipeline saved to {pipeline_path}")

    # 2. Model comparison JSON
    comparison_path = os.path.join(ARTIFACTS_DIR, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Model comparison saved to {comparison_path}")

    # 3. Champion metadata JSON
    champion_metadata = {
        "champion_model": champion_name,
        "selected_threshold": best_thresh,
        "test_metrics": comparison_results[champion_name]["test_metrics"],
        "validation_metrics": tuned_val_metrics,
        "class_imbalance": {
            "legitimate_rate": round(float((y_test == 0).mean()), 4),
            "fraud_rate": round(float((y_test == 1).mean()), 4)
        },
        "disclaimer": (
            "JobShield is an AI-assisted screening decision-support aid, not a legal finding or definitive accusation. "
            "Model predictions reflect statistical associations learned from training data."
        )
    }
    meta_path = os.path.join(ARTIFACTS_DIR, "champion_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(champion_metadata, f, indent=2)
    print(f"Champion metadata saved to {meta_path}")

    # 4. Error analysis JSON
    error_path = os.path.join(ARTIFACTS_DIR, "error_analysis.json")
    with open(error_path, "w") as f:
        json.dump(error_analysis, f, indent=2)
    print(f"Error analysis saved to {error_path}")

    # 5. Save Curated Demo Samples (2 Genuine, 2 Fraudulent, 1 Edge Case)
    curated_samples = [
        {
            "id": "sample_legit_1",
            "name": "Senior Full-Stack Engineer (Legitimate Enterprise)",
            "expected": "Likely Genuine",
            "title": "Senior Full-Stack Software Engineer",
            "location": "US, NY, New York",
            "department": "Engineering",
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "telecommuting": 0,
            "has_company_logo": 1,
            "has_questions": 1,
            "company_profile": "Datadog is the essential monitoring and security platform for cloud applications. We bring together data from servers, containers, databases, and third-party services to make your stack entirely observable, helping engineering teams resolve issues faster and maintain resilient systems worldwide.",
            "description": "We are seeking a Senior Full-Stack Software Engineer to build scalable, high-throughput cloud dashboards and microservices. You will collaborate with product designers, distributed systems architects, and site reliability engineers to deliver performant React and Go applications used by thousands of global customers.",
            "requirements": "5+ years of production experience with Python, Go, or TypeScript. In-depth understanding of PostgreSQL, Redis, REST/gRPC APIs, and containerized deployments using Docker and Kubernetes. Strong problem-solving abilities and track record of building production web services.",
            "benefits": "Competitive base salary, equity stock options, comprehensive health, dental, and vision insurance, 401(k) matching, and continuous learning stipend."
        },
        {
            "id": "sample_fraud_1",
            "name": "Urgent Data Entry Assistant (Phishing Scam with Upfront Fee)",
            "expected": "Potentially Fraudulent",
            "title": "URGENT Data Entry Clerk / Administrative Assistant Needed Immediately",
            "location": "US, TX, Austin",
            "department": "Admin",
            "employment_type": "Part-time",
            "required_experience": "Entry level",
            "required_education": "High School or equivalent",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "company_profile": "",
            "description": "URGENT HIRING! Work from home and make $500 per day! No experience required earn thousands weekly from your laptop. We will send you a cashier's check to purchase home office equipment and software. You must wire transfer the remaining balance back via Western Union or MoneyGram within 24 hours of receiving the check. Send your full name, bank account details, and copy of ID to hr.officer.career@gmail.com immediately.",
            "requirements": "Must have computer with internet access. No prior experience needed. Apply fast, positions filling fast!",
            "benefits": "Make unlimited money, flexible hours, daily payout guaranteed."
        },
        {
            "id": "sample_fraud_2",
            "name": "Cryptocurrency Investment Rep (Wire Transfer & Free Email)",
            "expected": "Potentially Fraudulent",
            "title": "Financial Account Manager - Crypto Payouts",
            "location": "US, CA, Los Angeles",
            "department": "Finance",
            "employment_type": "Full-time",
            "required_experience": "Not Applicable",
            "required_education": "Unspecified",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 0,
            "company_profile": "Global premier investment fund.",
            "description": "Earn $2500 weekly processing international bitcoin transactions. As an Account Representative, you will receive company client payments into your personal bank account or credit card, convert funds to cryptocurrency, and forward via wire transfer. Immediate start, no interview required. Please provide your social security number and bank details to start today.",
            "requirements": "Personal bank account in good standing. Fast response time.",
            "benefits": "High commission per transaction processed."
        },
        {
            "id": "sample_legit_2",
            "name": "Junior Marketing Specialist (Small Business Genuine)",
            "expected": "Likely Genuine",
            "title": "Digital Marketing Coordinator",
            "location": "US, IL, Chicago",
            "department": "Marketing",
            "employment_type": "Full-time",
            "required_experience": "Associate",
            "required_education": "Bachelor's Degree",
            "telecommuting": 1,
            "has_company_logo": 1,
            "has_questions": 1,
            "company_profile": "Lumina Creative is a boutique digital agency helping local retail brands expand their e-commerce footprint. Founded in 2018 in Chicago, we pride ourselves on transparent client partnerships and evidence-based SEO marketing.",
            "description": "We are looking for a creative, detail-oriented Digital Marketing Coordinator to manage social media campaigns, draft monthly newsletter content, and track Google Analytics conversion metrics.",
            "requirements": "Bachelor's degree in Marketing, Communications, or related discipline. 1-2 years experience managing Instagram and LinkedIn campaigns. Proficiency with Google Analytics and Mailchimp.",
            "benefits": "Hybrid work flexibility, health insurance, paid time off, and mentorship program."
        }
    ]
    samples_path = os.path.join(ARTIFACTS_DIR, "sample_test_cases.json")
    with open(samples_path, "w") as f:
        json.dump(curated_samples, f, indent=2)
    print(f"Curated sample test cases saved to {samples_path}")


if __name__ == "__main__":
    train_and_evaluate_all()
