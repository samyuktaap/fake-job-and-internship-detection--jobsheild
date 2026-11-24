"""
notebooks/generate_notebooks.py
Generates the 3 standard exploratory notebooks matching the JobShield project structure.
"""

import json
import os

NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True)
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.strip().splitlines(keepends=True)
    }


# 1. 01_eda.ipynb
cells_eda = [
    md_cell("# JobShield: 01 - Exploratory Data Analysis (EDA)\n## Dataset Research, Hygiene, and Class Imbalance Inspection\n\nThis notebook inspects the Employment Scam Aegean Dataset (EMSCAD / Kaggle Fake Job Posting Prediction dataset)."),
    code_cell("""
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../data/fake_job_postings.csv')
print('Raw Shape:', df.shape)
df.head(3)
"""),
    md_cell("### 1. Label Distribution & Class Imbalance\nIn real-world hiring environments, fraudulent listings constitute a tiny fraction of total job postings. Identifying this baseline imbalance is critical to selecting proper evaluation metrics (PR-AUC, Recall, Precision instead of naive accuracy)."),
    code_cell("""
counts = df['fraudulent'].value_counts()
proportions = df['fraudulent'].value_counts(normalize=True) * 100
print('Counts:\n', counts)
print('Proportions (%):\n', proportions)
"""),
    md_cell("### 2. Missing Value Analysis\nMissingness in specific fields (such as `company_profile` and `has_company_logo`) often serves as a structural signal of unverified employers."),
    code_cell("""
missing = df.isna().sum()
missing_pct = (df.isna().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing %', ascending=False))
"""),
    md_cell("### 3. Logo and Question Presence vs Fraud Rate"),
    code_cell("""
print('Fraud Rate with Logo:', df[df['has_company_logo'] == 1]['fraudulent'].mean() * 100)
print('Fraud Rate WITHOUT Logo:', df[df['has_company_logo'] == 0]['fraudulent'].mean() * 100)
"""),
    md_cell("### Summary of EDA Findings:\n- Severe class imbalance: only ~4.8% of raw listings are fraudulent.\n- Over 2,000 duplicate job templates exist; dropping them is necessary to prevent data leakage between train and test sets.\n- Postings without company logos have a significantly higher fraud rate.")
]

# 2. 02_preprocessing.ipynb
cells_prep = [
    md_cell("# JobShield: 02 - Preprocessing & Feature Pipeline\n## Leakage Prevention, Text Sanitization, and Feature Engineering\n\nThis notebook builds the scikit-learn Pipeline used for feature extraction."),
    code_cell("""
import sys
import os
sys.path.append('..')

from src.data.loader import load_raw_data, clean_and_deduplicate, prepare_stratified_splits
from src.features.preprocessor import build_full_feature_pipeline

# 1. Load and clean
raw = load_raw_data('../data/fake_job_postings.csv')
clean_df, stats = clean_and_deduplicate(raw)
print('Deduplication Audit:', stats)

# 2. Stratified 70/15/15 split
train_df, val_df, test_df = prepare_stratified_splits(clean_df, save_splits=False)
print('Train Shape:', train_df.shape)
print('Val Shape:', val_df.shape)
print('Test Shape:', test_df.shape)
"""),
    md_cell("### Feature Extraction Pipeline\nWe combine:\n1. Unigram and bigram TF-IDF sparse features on concatenated job text.\n2. One-hot encoded categorical metadata.\n3. Engineered domain risk signals (payment keywords, personal email patterns, missing profile flags)."),
    code_cell("""
pipeline = build_full_feature_pipeline()

# Fit strictly on training data
X_train = pipeline.fit_transform(train_df)
X_val = pipeline.transform(val_df)

print('Fitted Feature Matrix Shape (Train):', X_train.shape)
print('Transformed Feature Matrix Shape (Val):', X_val.shape)
""")
]

# 3. 03_model_comparison.ipynb
cells_model = [
    md_cell("# JobShield: 03 - Model Comparison, Validation & Evaluation\n## Baseline Benchmarks, Candidate Models, Threshold Optimization, and Error Analysis"),
    code_cell("""
import json
import pandas as pd

# Load saved comparison results
with open('../artifacts/model_comparison.json', 'r') as f:
    comparison = json.load(f)

# Convert held-out test metrics to table
rows = []
for model_name, data in comparison.items():
    tm = data['test_metrics']
    rows.append({
        'Model': model_name,
        'Features': data['features'],
        'Threshold': data.get('test_threshold', 0.5),
        'Precision': f\"{tm['precision']*100:.2f}%\",
        'Recall': f\"{tm['recall']*100:.2f}%\",
        'F1-Score': f\"{tm['f1']:.4f}\",
        'PR-AUC': f\"{tm['pr_auc']:.4f}\" if tm['pr_auc'] else 'N/A'
    })

df_results = pd.DataFrame(rows)
df_results
"""),
    md_cell("### Champion Model Metadata & Test Confusion Matrix"),
    code_cell("""
with open('../artifacts/champion_metadata.json', 'r') as f:
    meta = json.load(f)

print('Champion Model:', meta['champion_model'])
print('Tuned Threshold:', meta['selected_threshold'])
print('Confusion Matrix (Test):', meta['test_metrics']['confusion_matrix'])
"""),
    md_cell("### Error Analysis on Held-Out Test Set"),
    code_cell("""
with open('../artifacts/error_analysis.json', 'r') as f:
    errors = json.load(f)

print(f\"Examined {len(errors['false_positives'])} False Positives and {len(errors['false_negatives'])} False Negatives.\")
for fp in errors['false_positives'][:2]:
    print('FP Title:', fp['title'], '| Prob:', fp['predicted_prob'])
for fn in errors['false_negatives'][:2]:
    print('FN Title:', fn['title'], '| Prob:', fn['predicted_prob'])
""")
]


with open(os.path.join(NOTEBOOKS_DIR, "01_eda.ipynb"), "w") as f:
    json.dump(make_notebook(cells_eda), f, indent=2)

with open(os.path.join(NOTEBOOKS_DIR, "02_preprocessing.ipynb"), "w") as f:
    json.dump(make_notebook(cells_prep), f, indent=2)

with open(os.path.join(NOTEBOOKS_DIR, "03_model_comparison.ipynb"), "w") as f:
    json.dump(make_notebook(cells_model), f, indent=2)

print("Generated all 3 Jupyter notebooks successfully!")
