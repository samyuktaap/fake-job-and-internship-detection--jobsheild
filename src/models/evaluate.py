"""
src/models/evaluate.py
Evaluation metrics, confusion matrix computation, and error analysis.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes all essential evaluation metrics recommended by JobShield guide:
    - Precision (fraud class 1)
    - Recall (fraud class 1)
    - F1-score (fraud class 1)
    - PR-AUC (Average Precision)
    - ROC-AUC
    - Accuracy
    - Confusion Matrix [TN, FP, FN, TP]
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

    if y_proba is not None:
        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 4)
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception:
            metrics["pr_auc"] = None
            metrics["roc_auc"] = None
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.70
) -> Tuple[float, Dict[str, Any]]:
    """
    Scans candidate decision thresholds [0.05, 0.95] on validation data
    to balance Precision and Recall for the minority fraudulent class.
    
    Returns:
        best_threshold: Selected threshold.
        best_metrics: Metrics at that threshold.
    """
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.05, 0.90, 86)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # We prioritize maximizing F1-score subject to a healthy recall
        if rec >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(t), 3)
            best_metrics = calculate_metrics(y_true, y_pred, y_proba)

    # Fallback if no threshold met min_recall
    if best_f1 == -1.0:
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = round(float(t), 3)
                best_metrics = calculate_metrics(y_true, y_pred, y_proba)

    return best_threshold, best_metrics


def perform_error_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    max_examples: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts false positive and false negative examples to understand model error modes.
    """
    false_positives = []
    false_negatives = []

    for idx, (yt, yp, prob) in enumerate(zip(y_true, y_pred, y_proba)):
        row = df.iloc[idx]
        item = {
            "index": int(idx),
            "job_id": int(row.get("job_id", idx)),
            "title": str(row.get("title", "Untitled")),
            "company_profile": str(row.get("company_profile", ""))[:200] + ("..." if len(str(row.get("company_profile", ""))) > 200 else ""),
            "description": str(row.get("description", ""))[:300] + ("..." if len(str(row.get("description", ""))) > 300 else ""),
            "requirements": str(row.get("requirements", ""))[:200] + ("..." if len(str(row.get("requirements", ""))) > 200 else ""),
            "has_company_logo": int(row.get("has_company_logo", 0)),
            "telecommuting": int(row.get("telecommuting", 0)),
            "predicted_prob": round(float(prob), 4),
            "true_label": int(yt),
            "pred_label": int(yp)
        }

        if yt == 0 and yp == 1 and len(false_positives) < max_examples:
            # False Positive: Legitimate job flagged as fraudulent
            item["error_explanation"] = (
                "Legitimate listing falsely flagged: Often occurs when genuine small employers "
                "lack a company profile, logo, or use informal remote hiring language."
            )
            false_positives.append(item)
        elif yt == 1 and yp == 0 and len(false_negatives) < max_examples:
            # False Negative: Fraudulent job missed
            item["error_explanation"] = (
                "Fraudulent listing missed: Scam posting closely mimicked legitimate corporate "
                "job descriptions or plagiarized a real enterprise's specifications."
            )
            false_negatives.append(item)

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
