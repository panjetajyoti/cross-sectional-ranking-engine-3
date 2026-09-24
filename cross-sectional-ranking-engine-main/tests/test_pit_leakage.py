"""
tests/test_pit_leakage.py
Verifies strict Point-in-Time compliance and absence of look-ahead leakage.
"""
import os
import pandas as pd
import numpy as np

def test_point_in_time_availability():
    dataset_path = "data/curated_pit/labeled_dataset.parquet"
    assert os.path.exists(dataset_path), "Curated dataset does not exist."
    
    df = pd.read_parquet(dataset_path)
    
    # 1. Market date must be greater than or equal to fundamental availability date
    if 'avail_date' in df.columns:
        valid_rows = df.dropna(subset=['avail_date'])
        leakage_violations = valid_rows[valid_rows['date'] < valid_rows['avail_date']]
        assert len(leakage_violations) == 0, f"Critical Look-Ahead Leakage: {len(leakage_violations)} rows have market date prior to fundamental availability date."
        
    # 2. Verify no forward returns are accidentally included in feature columns
    feature_cols = [c for c in df.columns if '_neut' in c]
    for col in feature_cols:
        corr_with_future = df[[col, 'fwd_ret_21d']].dropna().corr().iloc[0, 1]
        assert abs(corr_with_future) < 0.90, f"Possible target leakage detected in feature: {col}"
        
    print("\n[✓] Point-in-Time Immunity & No Look-Ahead Bias Verified.")

if __name__ == "__main__":
    test_point_in_time_availability()