"""
src/models/audit_evaluation.py
Optimized ML audit and benchmarking script for JobShield 2.0.
Performs 5-fold Stratified Cross-Validation on development data and
evaluates all candidate models (including XGBoost) on the untouched test set.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from src.features.preprocessor import build_text_only_pipeline, build_full_feature_pipeline
from src.models.evaluate import calculate_metrics, find_optimal_threshold, perform_error_analysis

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")


def run_comprehensive_audit():
    print("Loading data splits...", flush=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    y_dev = dev_df["fraudulent"].values
    y_train = train_df["fraudulent"].values
    y_val = val_df["fraudulent"].values
    y_test = test_df["fraudulent"].values

    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"Data ready: Dev={len(dev_df)}, Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}", flush=True)

    # Pre-extract text-only and full features to accelerate 5-fold CV
    print("Precomputing feature representations for fast CV...", flush=True)
    text_pipe = build_text_only_pipeline()
    full_pipe = build_full_feature_pipeline()

    # Fit feature pipelines strictly on dev_df for CV
    X_text_dev = text_pipe.fit_transform(dev_df)
    X_full_dev = full_pipe.fit_transform(dev_df)
    print(f"Feature matrices ready: Text={X_text_dev.shape}, Full={X_full_dev.shape}", flush=True)

    # Fit feature pipelines strictly on train_df for final test evaluation (leakage-free)
    print("Precomputing feature representations on train_df for test evaluation...", flush=True)
    text_pipe_train = build_text_only_pipeline()
    full_pipe_train = build_full_feature_pipeline()

    X_text_train = text_pipe_train.fit_transform(train_df)
    X_text_val = text_pipe_train.transform(val_df)
    X_text_test = text_pipe_train.transform(test_df)

    X_full_train = full_pipe_train.fit_transform(train_df)
    X_full_val = full_pipe_train.transform(val_df)
    X_full_test = full_pipe_train.transform(test_df)

    # Model definitions: (name, feature_type, classifier_factory)
    model_configs = [
        ("DummyClassifier (Majority)", "text", lambda: DummyClassifier(strategy="most_frequent")),
        ("Multinomial Naive Bayes", "text", lambda: MultinomialNB(alpha=0.1)),
        ("Logistic Regression (Balanced)", "full", lambda: LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="liblinear", random_state=42)),
        ("Linear SVM (Calibrated)", "full", lambda: CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", max_iter=2000, random_state=42, dual="auto"),
            cv=3
        )),
        ("Random Forest (Balanced)", "full", lambda: RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("XGBoost (scale_pos_weight)", "full", lambda: xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ]

    # 1. 5-Fold Stratified Cross-Validation on Development Split
    print("\n--- Running 5-Fold Stratified Cross-Validation ---", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}

    for name, ftype, clf_factory in model_configs:
        print(f"Running 5-Fold CV for [{name}]...", flush=True)
        X_mat = X_text_dev if ftype == "text" else X_full_dev

        f1_scores = []
        prec_scores = []
        rec_scores = []
        pr_auc_scores = []
        roc_auc_scores = []

        for train_idx, val_idx in skf.split(X_mat, y_dev):
            clf = clf_factory()
            clf.fit(X_mat[train_idx], y_dev[train_idx])

            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_mat[val_idx])[:, 1]
                pred = (proba >= 0.5).astype(int)
            else:
                pred = clf.predict(X_mat[val_idx])
                proba = None

            m = calculate_metrics(y_dev[val_idx], pred, proba)
            f1_scores.append(m["f1"])
            prec_scores.append(m["precision"])
            rec_scores.append(m["recall"])
            if m["pr_auc"] is not None:
                pr_auc_scores.append(m["pr_auc"])
            if m["roc_auc"] is not None:
                roc_auc_scores.append(m["roc_auc"])

        cv_summary[name] = {
            "cv_f1_mean": round(float(np.mean(f1_scores)), 4),
            "cv_f1_std": round(float(np.std(f1_scores)), 4),
            "cv_precision_mean": round(float(np.mean(prec_scores)), 4),
            "cv_recall_mean": round(float(np.mean(rec_scores)), 4),
            "cv_pr_auc_mean": round(float(np.mean(pr_auc_scores)), 4) if pr_auc_scores else None,
            "cv_roc_auc_mean": round(float(np.mean(roc_auc_scores)), 4) if roc_auc_scores else None,
        }
        print(f"  [{name}] CV F1: {cv_summary[name]['cv_f1_mean']:.4f} (+/- {cv_summary[name]['cv_f1_std']:.4f}) | PR-AUC: {cv_summary[name]['cv_pr_auc_mean']}", flush=True)

    # 2. Final Training & Evaluation on Untouched Held-Out Test Set
    print("\n--- Training on train_df & Evaluating on Untouched Held-Out Test Set ---", flush=True)
    test_results = []
    error_analysis_data = {}

    for name, ftype, clf_factory in model_configs:
        print(f"Evaluating [{name}] on held-out test set...", flush=True)
        if ftype == "text":
            X_tr, X_v, X_te = X_text_train, X_text_val, X_text_test
        else:
            X_tr, X_v, X_te = X_full_train, X_full_val, X_full_test

        clf = clf_factory()
        clf.fit(X_tr, y_train)

        # Threshold selection on validation split
        if hasattr(clf, "predict_proba"):
            val_proba = clf.predict_proba(X_v)[:, 1]
            test_proba = clf.predict_proba(X_te)[:, 1]
            best_thresh, _ = find_optimal_threshold(y_val, val_proba, min_recall=0.70)
        else:
            val_proba = None
            test_proba = None
            best_thresh = 0.5

        if test_proba is not None:
            test_pred = (test_proba >= best_thresh).astype(int)
            test_m = calculate_metrics(y_test, test_pred, test_proba)
        else:
            test_pred = clf.predict(X_te)
            test_m = calculate_metrics(y_test, test_pred, None)
            best_thresh = 0.5

        if name == "Logistic Regression (Balanced)":
            error_analysis_data = perform_error_analysis(test_df, y_test, test_pred, test_proba, max_examples=10)

        record = {
            "Model": name,
            "Features": "TF-IDF text" if ftype == "text" else "TF-IDF + Categorical + Risk Signals",
            "CV_F1_Mean": cv_summary[name]["cv_f1_mean"],
            "CV_F1_Std": cv_summary[name]["cv_f1_std"],
            "CV_PR_AUC": cv_summary[name]["cv_pr_auc_mean"],
            "Val_Optimal_Threshold": best_thresh,
            "Test_Threshold": best_thresh,
            "Test_Accuracy": test_m["accuracy"],
            "Test_Precision": test_m["precision"],
            "Test_Recall": test_m["recall"],
            "Test_F1": test_m["f1"],
            "Test_PR_AUC": test_m["pr_auc"],
            "Test_ROC_AUC": test_m["roc_auc"],
            "TN": test_m["true_negatives"],
            "FP": test_m["false_positives"],
            "FN": test_m["false_negatives"],
            "TP": test_m["true_positives"],
        }
        test_results.append(record)
        print(f"  [{name}] (thresh={best_thresh}) -> Acc: {record['Test_Accuracy']:.4f} | Prec: {record['Test_Precision']:.4f} | Rec: {record['Test_Recall']:.4f} | F1: {record['Test_F1']:.4f} | PR-AUC: {record['Test_PR_AUC']}", flush=True)
        print(f"      CM: TN={record['TN']}, FP={record['FP']}, FN={record['FN']}, TP={record['TP']}", flush=True)

    # Save MODEL_COMPARISON.csv at project root
    df_csv = pd.DataFrame(test_results)
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "MODEL_COMPARISON.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"\nSaved MODEL_COMPARISON.csv to {csv_path}", flush=True)

    # Save detailed audit JSON
    audit_json_path = os.path.join(ARTIFACTS_DIR, "audit_metrics.json")
    with open(audit_json_path, "w") as f:
        json.dump({
            "cross_validation": cv_summary,
            "test_benchmarks": test_results,
            "error_analysis": error_analysis_data
        }, f, indent=2)
    print(f"Saved audit metrics to {audit_json_path}", flush=True)
    print("ALL AUDIT BENCHMARKS COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    run_comprehensive_audit()
