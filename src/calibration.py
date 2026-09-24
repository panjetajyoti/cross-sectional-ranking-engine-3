"""
src/calibration.py
Calibrates raw ranking scores into true probabilities P(Outperform > median).
Implements Platt Scaling (Sigmoid) and Isotonic Regression.
Computes Expected Calibration Error (ECE) and plots reliability curves.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    Measures difference between predicted confidence and empirical frequency.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    n_samples = len(y_true)
    
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / n_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def run_calibration_pipeline(oos_path="outputs/shortlists/oos_ranked_predictions.parquet", out_dir="outputs"):
    """
    Loads Out-Of-Fold predictions, trains calibrators, evaluates ECE,
    and updates shortlisted data with calibrated probabilities.
    """
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    
    print(f"[*] Loading OOS predictions from {oos_path}...")
    df = pd.read_parquet(oos_path)
    
    X_scores = df[['ranker_score']].values
    y_true = df['binary_outperform'].values
    
    # 1. Platt Scaling (Logistic Sigmoid)
    platt_model = LogisticRegression(C=1.0, solver='lbfgs')
    platt_model.fit(X_scores, y_true)
    df['prob_platt'] = platt_model.predict_proba(X_scores)[:, 1]
    
    # 2. Isotonic Regression (Non-parametric)
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(X_scores.ravel(), y_true)
    df['prob_isotonic'] = isotonic_model.predict(X_scores.ravel())
    
    # Choose Platt as primary calibrated propensity
    df['calibrated_propensity'] = df['prob_platt']
    
    # 3. Calculate ECE
    ece_raw = compute_ece(y_true, (df['percentile_rank']).values)
    ece_platt = compute_ece(y_true, df['prob_platt'].values)
    ece_isotonic = compute_ece(y_true, df['prob_isotonic'].values)
    
    print(f"[+] Raw Percentile ECE   : {ece_raw:.4f}")
    print(f"[+] Platt Scaling ECE    : {ece_platt:.4f}")
    print(f"[+] Isotonic Scaling ECE : {ece_isotonic:.4f}")
    
    metrics = {
        "ece_uncalibrated": round(ece_raw, 4),
        "ece_platt": round(ece_platt, 4),
        "ece_isotonic": round(ece_isotonic, 4)
    }
    
    with open(os.path.join(out_dir, "metrics", "calibration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 4. Generate Reliability Diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, df['percentile_rank'].values, n_bins=10)
    frac_pos_platt, mean_pred_platt = calibration_curve(y_true, df['prob_platt'].values, n_bins=10)
    frac_pos_iso, mean_pred_iso = calibration_curve(y_true, df['prob_isotonic'].values, n_bins=10)
    
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y=x)")
    ax.plot(mean_pred_raw, frac_pos_raw, "s-", color="red", label=f"Uncalibrated (ECE: {ece_raw:.3f})")
    ax.plot(mean_pred_platt, frac_pos_platt, "o-", color="blue", label=f"Platt Scaling (ECE: {ece_platt:.3f})")
    ax.plot(mean_pred_iso, frac_pos_iso, "^-", color="green", label=f"Isotonic (ECE: {ece_isotonic:.3f})")
    
    ax.set_title("Reliability Diagram - Probability Calibration", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Out-Performance Probability", fontsize=10)
    ax.set_ylabel("Empirical Fraction of Winners", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(out_dir, "figures", "reliability_diagram.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[✓] Reliability plot saved to {fig_path}")
    
    # Save back calibrated predictions
    df.to_parquet(oos_path, index=False)
    print(f"[✓] Calibrated probabilities saved back into {oos_path}")
    return df

if __name__ == "__main__":
    run_calibration_pipeline()