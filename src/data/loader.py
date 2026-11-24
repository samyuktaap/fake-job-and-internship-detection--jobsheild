"""
src/data/loader.py
Data loading, cleaning, deduplication, and stratified split module for JobShield.
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "fake_job_postings.csv")


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw EMSCAD fake job postings dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)


def clean_and_deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean dataset and remove duplicates to avoid data leakage between splits.
    
    Returns:
        cleaned_df: Deduplicated and sanitized DataFrame.
        stats: Dictionary containing cleaning audit metrics.
    """
    initial_rows = len(df)
    
    # Text fields where NaN should be converted to empty string
    text_cols = ["title", "company_profile", "description", "requirements", "benefits", "location", "department"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
            
    # Structured categorical fields
    cat_cols = ["employment_type", "required_experience", "required_education", "industry", "function"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unspecified").astype(str)
            
    # Binary indicator fields (fill NaN with 0)
    binary_cols = ["telecommuting", "has_company_logo", "has_questions"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Ensure target is integer
    df["fraudulent"] = df["fraudulent"].fillna(0).astype(int)

    # Deduplicate 1: Exact row duplicates (excluding job_id)
    content_cols = [c for c in df.columns if c != "job_id"]
    exact_duplicates = int(df.duplicated(subset=content_cols).sum())
    df_no_exact = df.drop_duplicates(subset=content_cols).copy()
    
    # Deduplicate 2: Text duplicates on (title, company_profile, description)
    # This prevents the same job template (posted in multiple cities) from leaking across train/test splits.
    text_dup_subset = ["title", "company_profile", "description"]
    text_duplicates = int(df_no_exact.duplicated(subset=text_dup_subset).sum())
    cleaned_df = df_no_exact.drop_duplicates(subset=text_dup_subset).reset_index(drop=True)
    
    stats = {
        "initial_rows": initial_rows,
        "exact_duplicates_removed": exact_duplicates,
        "text_template_duplicates_removed": text_duplicates,
        "final_cleaned_rows": len(cleaned_df),
        "legitimate_count": int((cleaned_df["fraudulent"] == 0).sum()),
        "fraudulent_count": int((cleaned_df["fraudulent"] == 1).sum()),
        "fraud_prevalence_pct": round(float(cleaned_df["fraudulent"].mean()) * 100, 2),
    }
    return cleaned_df, stats


def prepare_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a 3-way stratified split: Train (70%), Validation (15%), Test (15%).
    Saves the splits to disk for reproducible experimentation.
    """
    y = df["fraudulent"]
    
    # First split off the test set (15%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Relative validation size from remaining 85%: 0.15 / 0.85 approx 0.1765
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["fraudulent"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if save_splits:
        train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
        test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df, stats = clean_and_deduplicate(raw_df)
    print("Deduplication and Cleaning Stats:", stats)
    train_df, val_df, test_df = prepare_stratified_splits(cleaned_df)
    print(f"Splits prepared:\n  Train: {train_df.shape} (Fraud: {train_df['fraudulent'].sum()})\n  Val:   {val_df.shape} (Fraud: {val_df['fraudulent'].sum()})\n  Test:  {test_df.shape} (Fraud: {test_df['fraudulent'].sum()})")
