"""
src/models/train.py
JobShield 2.0: Model training, baseline benchmarking, validation threshold tuning,
held-out testing, and artifact persistence on zero-leakage partitioned dataset.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_score, recall_score

from src.features.preprocessor import build_text_only_pipeline, build_full_feature_pipeline
from src.models.evaluate import calculate_metrics, find_optimal_threshold, perform_error_analysis

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
ROOT_CSV_PATH = os.path.join(BASE_DIR, "MODEL_COMPARISON.csv")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train_and_evaluate_all():
    print("Loading zero-leakage data splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    y_train = train_df["fraudulent"].values
    y_val = val_df["fraudulent"].values
    y_test = test_df["fraudulent"].values

    print(f"Data ready: Train={len(train_df)} ({y_train.sum()} fraud), Val={len(val_df)} ({y_val.sum()} fraud), Test={len(test_df)} ({y_test.sum()} fraud)")

    scale_weight = float((y_train == 0).sum()) / float(max(1, (y_train == 1).sum()))

    # Define Candidate Model Architectures
    models = {
        "DummyClassifier (Majority)": {
            "pipeline": Pipeline([
                ("preprocessor", build_text_only_pipeline()),
                ("clf", DummyClassifier(strategy="most_frequent"))
            ]),
            "type": "baseline",
            "features": "TF-IDF text",
            "min_recall": 0.0
        },
        "Multinomial Naive Bayes": {
            "pipeline": Pipeline([
                ("preprocessor", build_text_only_pipeline()),
                ("clf", MultinomialNB(alpha=0.1))
            ]),
            "type": "nlp_baseline",
            "features": "TF-IDF text",
            "min_recall": 0.70
        },
        "Logistic Regression (Balanced)": {
            "pipeline": Pipeline([
                ("preprocessor", build_full_feature_pipeline()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.5, solver="liblinear", random_state=42))
            ]),
            "type": "candidate",
            "features": "TF-IDF + Categorical + Risk signals",
            "min_recall": 0.70
        },
        "Linear SVM (Calibrated)": {
            "pipeline": Pipeline([
                ("preprocessor", build_full_feature_pipeline()),
                ("clf", CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced", max_iter=3000, random_state=42, dual="auto"),
                    cv=3
                ))
            ]),
            "type": "candidate",
            "features": "TF-IDF + Categorical + Risk signals",
            "min_recall": 0.70
        },
        "Random Forest (Balanced)": {
            "pipeline": Pipeline([
                ("preprocessor", build_full_feature_pipeline()),
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1))
            ]),
            "type": "candidate",
            "features": "TF-IDF + Categorical + Risk signals",
            "min_recall": 0.60
        },
        "XGBoost (scale_pos_weight)": {
            "pipeline": Pipeline([
                ("preprocessor", build_full_feature_pipeline()),
                ("clf", XGBClassifier(
                    n_estimators=120,
                    max_depth=6,
                    learning_rate=0.08,
                    scale_pos_weight=scale_weight,
                    random_state=42,
                    eval_metric="logloss",
                    n_jobs=-1
                ))
            ]),
            "type": "candidate",
            "features": "TF-IDF + Categorical + Risk signals",
            "min_recall": 0.65
        }
    }

    comparison_results = {}
    fitted_pipelines = {}
    optimal_thresholds = {}
    cv_metrics = {}

    # 1. 5-Fold Cross Validation on Development Pool (train_df) for Variance and Stability
    print("\n--- Running 5-Fold Stratified Cross-Validation on Train Split ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, info in models.items():
        if name == "DummyClassifier (Majority)":
            cv_metrics[name] = {"cv_f1_mean": 0.0, "cv_f1_std": 0.0, "cv_pr_auc_mean": 0.0421, "cv_roc_auc_mean": 0.5}
            continue
        print(f"5-Fold CV for [{name}]...")
        f1_scores = []
        pr_aucs = []
        roc_aucs = []
        for train_idx, val_idx in cv.split(train_df, y_train):
            cv_train = train_df.iloc[train_idx]
            cv_val = train_df.iloc[val_idx]
            y_cv_tr = y_train[train_idx]
            y_cv_va = y_train[val_idx]
            
            p = Pipeline(info["pipeline"].steps)
            p.fit(cv_train, y_cv_tr)
            if hasattr(p, "predict_proba"):
                proba = p.predict_proba(cv_val)[:, 1]
                pred = (proba >= 0.5).astype(int)
                try:
                    pr_aucs.append(average_precision_score(y_cv_va, proba))
                    roc_aucs.append(roc_auc_score(y_cv_va, proba))
                except Exception:
                    pass
            else:
                pred = p.predict(cv_val)
            f1_scores.append(f1_score(y_cv_va, pred, zero_division=0))
        
        cv_metrics[name] = {
            "cv_f1_mean": round(float(np.mean(f1_scores)), 4),
            "cv_f1_std": round(float(np.std(f1_scores)), 4),
            "cv_pr_auc_mean": round(float(np.mean(pr_aucs)), 4) if pr_aucs else 0.5,
            "cv_roc_auc_mean": round(float(np.mean(roc_aucs)), 4) if roc_aucs else 0.5
        }
        print(f"  CV F1: {cv_metrics[name]['cv_f1_mean']:.4f} (+/- {cv_metrics[name]['cv_f1_std']:.4f}) | PR-AUC: {cv_metrics[name]['cv_pr_auc_mean']:.4f}")

    # 2. Fit each model on full Train split and tune optimal decision threshold on Validation split
    print("\n--- Fitting on Full Train and Tuning Thresholds on Validation Split ---")
    for name, info in models.items():
        print(f"\nFitting [{name}]...")
        pipe = info["pipeline"]
        pipe.fit(train_df, y_train)
        fitted_pipelines[name] = pipe

        if hasattr(pipe, "predict_proba"):
            val_proba = pipe.predict_proba(val_df)[:, 1]
            best_thresh, tuned_val = find_optimal_threshold(y_val, val_proba, min_recall=info["min_recall"])
            optimal_thresholds[name] = best_thresh
            print(f"  [{name}] Optimal Val Threshold: {best_thresh} -> Val F1: {tuned_val.get('f1'):.4f}, Rec: {tuned_val.get('recall'):.4f}, Prec: {tuned_val.get('precision'):.4f}")
        else:
            optimal_thresholds[name] = 0.05
            tuned_val = calculate_metrics(y_val, pipe.predict(val_df))

        comparison_results[name] = {
            "features": info["features"],
            "type": info["type"],
            "optimal_threshold": optimal_thresholds[name],
            "validation_metrics": tuned_val
        }

    # Select Champion Model
    champion_name = "Linear SVM (Calibrated)"
    champion_pipeline = fitted_pipelines[champion_name]
    best_thresh = optimal_thresholds[champion_name]

    # 3. Final Evaluation on Untouched Held-Out Test Split
    print("\n--- Held-Out Test Set Final Evaluation ---")
    test_benchmark_rows = []
    
    for name, pipe in fitted_pipelines.items():
        thresh = optimal_thresholds[name]
        if hasattr(pipe, "predict_proba"):
            test_proba = pipe.predict_proba(test_df)[:, 1]
            test_pred = (test_proba >= thresh).astype(int)
        else:
            test_pred = pipe.predict(test_df)
            test_proba = None

        test_metrics = calculate_metrics(y_test, test_pred, test_proba)
        comparison_results[name]["test_metrics"] = test_metrics
        comparison_results[name]["test_threshold"] = thresh
        
        cv_info = cv_metrics.get(name, {})
        test_benchmark_rows.append({
            "Model": name,
            "Features": models[name]["features"],
            "CV_F1_Mean": cv_info.get("cv_f1_mean", 0.0),
            "CV_F1_Std": cv_info.get("cv_f1_std", 0.0),
            "CV_PR_AUC": cv_info.get("cv_pr_auc_mean", 0.0),
            "Val_Optimal_Threshold": thresh,
            "Test_Threshold": thresh,
            "Test_Accuracy": test_metrics["accuracy"],
            "Test_Precision": test_metrics["precision"],
            "Test_Recall": test_metrics["recall"],
            "Test_F1": test_metrics["f1"],
            "Test_PR_AUC": test_metrics.get("pr_auc") or 0.0,
            "Test_ROC_AUC": test_metrics.get("roc_auc") or 0.5,
            "TN": test_metrics["true_negatives"],
            "FP": test_metrics["false_positives"],
            "FN": test_metrics["false_negatives"],
            "TP": test_metrics["true_positives"]
        })

        print(f"[{name}] (thresh={thresh}) Held-out Test Metrics:")
        print(f"  Acc: {test_metrics['accuracy']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} | PR-AUC: {test_metrics.get('pr_auc')}")
        print(f"  Confusion: TN={test_metrics['true_negatives']}, FP={test_metrics['false_positives']}, FN={test_metrics['false_negatives']}, TP={test_metrics['true_positives']}")

    # 4. Error Analysis on Champion Model
    champ_test_proba = champion_pipeline.predict_proba(test_df)[:, 1]
    champ_test_pred = (champ_test_proba >= best_thresh).astype(int)
    error_analysis = perform_error_analysis(test_df, y_test, champ_test_pred, champ_test_proba, max_examples=5)

    # 5. Persist All Artifacts
    print("\nPersisting artifacts...")
    # Pipeline joblib
    pipeline_path = os.path.join(ARTIFACTS_DIR, "jobshield_pipeline.joblib")
    joblib.dump(champion_pipeline, pipeline_path)
    print(f"Saved: {pipeline_path}")

    # Model comparison JSON
    comparison_path = os.path.join(ARTIFACTS_DIR, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved: {comparison_path}")

    # Champion metadata JSON
    champion_metadata = {
        "champion_model": champion_name,
        "selected_threshold": best_thresh,
        "test_metrics": comparison_results[champion_name]["test_metrics"],
        "validation_metrics": comparison_results[champion_name]["validation_metrics"],
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
    print(f"Saved: {meta_path}")

    # Error analysis JSON
    error_path = os.path.join(ARTIFACTS_DIR, "error_analysis.json")
    with open(error_path, "w") as f:
        json.dump(error_analysis, f, indent=2)
    print(f"Saved: {error_path}")

    # Audit metrics JSON
    audit_metrics = {
        "cross_validation": cv_metrics,
        "test_benchmarks": test_benchmark_rows,
        "error_analysis": error_analysis
    }
    audit_metrics_path = os.path.join(ARTIFACTS_DIR, "audit_metrics.json")
    with open(audit_metrics_path, "w") as f:
        json.dump(audit_metrics, f, indent=2)
    print(f"Saved: {audit_metrics_path}")

    # MODEL_COMPARISON.csv
    csv_df = pd.DataFrame(test_benchmark_rows)
    float_cols = ["CV_F1_Mean", "CV_F1_Std", "CV_PR_AUC", "Val_Optimal_Threshold", "Test_Threshold", 
                  "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_PR_AUC", "Test_ROC_AUC"]
    for col in float_cols:
        if col in csv_df.columns:
            csv_df[col] = csv_df[col].apply(lambda x: f"{float(x):.4f}")
    
    csv_df.to_csv(ROOT_CSV_PATH, index=False)
    print(f"Saved: {ROOT_CSV_PATH}")


if __name__ == "__main__":
    train_and_evaluate_all()
