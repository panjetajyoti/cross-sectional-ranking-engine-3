"""
tests/test_calibration.py
Validates probability calibration properties, bounds, and Expected Calibration Error (ECE).
"""
import os
import json
import pandas as pd
import numpy as np

def test_calibrated_probabilities():
    shortlist_path = "outputs/shortlists/oos_ranked_predictions.parquet"
    assert os.path.exists(shortlist_path), "OOS predictions dataset missing."
    
    df = pd.read_parquet(shortlist_path)
    assert 'calibrated_propensity' in df.columns, "calibrated_propensity column missing."
    
    probs = df['calibrated_propensity'].dropna()
    
    # Check probability bounds [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Calibrated probabilities violate [0, 1] mathematical range."
    
    # Verify calibration metrics file exists and ECE is acceptable
    metrics_path = "outputs/metrics/calibration_metrics.json"
    assert os.path.exists(metrics_path), "Calibration metrics JSON missing."
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    assert "ece_platt" in metrics, "ECE metric missing."
    assert metrics["ece_platt"] < 0.15, f"Calibration Error too high: {metrics['ece_platt']}"
    
    print("\n[✓] Probability Calibration Bounds & ECE Verified.")

if __name__ == "__main__":
    test_calibrated_probabilities()