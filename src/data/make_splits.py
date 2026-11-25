"""
src/data/make_splits.py
Deduplicates raw EMSCAD job postings strictly on normalized description body
to eliminate near-duplicate description leakage, then generates stratified
70% Train, 15% Validation, and 15% Held-Out Test splits.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PATH = os.path.join(DATA_DIR, "fake_job_postings.csv")


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def make_zero_leakage_splits(random_state: int = 42):
    print(f"Loading raw dataset from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df.shape}")

    # Normalize description for deduplication
    df["norm_desc"] = df["description"].apply(normalize_text)
    
    # Filter out empty or negligible descriptions (<30 chars)
    df_valid = df[df["norm_desc"].str.len() > 30].copy()
    
    # Remove conflicting descriptions (same description labeled both fraud and legit)
    conflicts = df_valid.groupby("norm_desc")["fraudulent"].nunique()
    conflicts = conflicts[conflicts > 1].index
    if len(conflicts) > 0:
        print(f"Removing {len(conflicts)} ambiguous descriptions with conflicting labels...")
        df_valid = df_valid[~df_valid["norm_desc"].isin(conflicts)]
    
    # Deduplicate strictly on normalized description body
    df_dedup = df_valid.drop_duplicates(subset=["norm_desc"]).copy()
    df_dedup = df_dedup.drop(columns=["norm_desc"]).reset_index(drop=True)
    
    fraud_count = int(df_dedup["fraudulent"].sum())
    total_count = len(df_dedup)
    print(f"Deduplicated dataset: {total_count} listings ({fraud_count} fraudulent, {fraud_count/total_count*100:.2f}%)")

    # Stratified 70 / 15 / 15 Split
    train_df, temp_df = train_test_split(
        df_dedup,
        test_size=0.30,
        random_state=random_state,
        stratify=df_dedup["fraudulent"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["fraudulent"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train split: {len(train_df)} samples ({train_df['fraudulent'].sum()} fraud, {train_df['fraudulent'].mean()*100:.2f}%)")
    print(f"Val split:   {len(val_df)} samples ({val_df['fraudulent'].sum()} fraud, {val_df['fraudulent'].mean()*100:.2f}%)")
    print(f"Test split:  {len(test_df)} samples ({test_df['fraudulent'].sum()} fraud, {test_df['fraudulent'].mean()*100:.2f}%)")

    # Verify zero overlap
    train_desc = set(train_df["description"].apply(normalize_text))
    val_desc = set(val_df["description"].apply(normalize_text))
    test_desc = set(test_df["description"].apply(normalize_text))

    assert len(train_desc.intersection(val_desc)) == 0, "Train-Val description leakage detected!"
    assert len(train_desc.intersection(test_desc)) == 0, "Train-Test description leakage detected!"
    assert len(val_desc.intersection(test_desc)) == 0, "Val-Test description leakage detected!"
    print("VERIFIED: Zero description text leakage across train, val, and test splits.")

    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    make_zero_leakage_splits()
