"""
src/conformal.py
Implements Mondrian (group-conditional) Conformal Prediction wrappers.
Provides statistical coverage guarantees for out-performer stock selection.
"""

import os
import json
import pandas as pd
import numpy as np

def compute_mondrian_conformal_thresholds(df_calib: pd.DataFrame, alpha: float = 0.10) -> dict:
    """
    Calculates non-conformity score quantile per sector (Mondrian conditioning).
    Non-conformity score s_i = 1 - P(outperform) for positive class.
    Coverage level = 1 - alpha (e.g., 90%).
    """
    thresholds = {}
    
    # Global fallback quantile
    global_scores = 1.0 - df_calib.loc[df_calib['binary_outperform'] == 1, 'calibrated_propensity']
    if len(global_scores) > 0:
        global_q = float(np.quantile(global_scores, 1.0 - alpha))
    else:
        global_q = 0.50
        
    for sector, grp in df_calib.groupby('sector'):
        # Calibration non-conformity on true winners
        true_winners = grp[grp['binary_outperform'] == 1]
        
        if len(true_winners) >= 10:
            scores = 1.0 - true_winners['calibrated_propensity']
            # (1 - alpha) conformal quantile
            q_val = float(np.quantile(scores, 1.0 - alpha))
            thresholds[sector] = round(q_val, 4)
        else:
            thresholds[sector] = round(global_q, 4)
            
    thresholds['default'] = round(global_q, 4)
    return thresholds

def apply_conformal_selection(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Labels each stock prediction with a conformal inclusion flag.
    If 1 - P(outperform) <= q_sector, the stock is included in the 90% confidence set.
    """
    df = df.copy()
    conformal_included = []
    
    for _, row in df.iterrows():
        sec = row.get('sector', 'default')
        q_th = thresholds.get(sec, thresholds['default'])
        
        non_conformity = 1.0 - row['calibrated_propensity']
        is_selected = int(non_conformity <= q_th)
        conformal_included.append(is_selected)
        
    df['conformal_selected_90'] = conformal_included
    return df

def run_conformal_pipeline(oos_path="outputs/shortlists/oos_ranked_predictions.parquet", out_dir="outputs"):
    """Loads calibrated predictions, computes sector conformal cutoffs, and saves shortlist."""
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "shortlists"), exist_ok=True)
    
    print(f"[*] Loading data from {oos_path}...")
    df = pd.read_parquet(oos_path)
    
    # Split historical data into calibration half and evaluation half
    dates = np.sort(df['date'].unique())
    split_idx = len(dates) // 2
    calib_dates = dates[:split_idx]
    eval_dates = dates[split_idx:]
    
    df_calib = df[df['date'].isin(calib_dates)].copy()
    df_eval = df[df['date'].isin(eval_dates)].copy()
    
    print(f"[*] Computing Mondrian sector-wise conformal thresholds (alpha=0.10)...")
    thresholds = compute_mondrian_conformal_thresholds(df_calib, alpha=0.10)
    
    # Apply to full dataset
    df_conformal = apply_conformal_selection(df, thresholds)
    
    # Check empirical coverage on eval set
    eval_conformal = df_conformal[df_conformal['date'].isin(eval_dates)]
    true_eval_winners = eval_conformal[eval_conformal['binary_outperform'] == 1]
    
    empirical_coverage = float(true_eval_winners['conformal_selected_90'].mean())
    avg_set_size = float(eval_conformal.groupby('date')['conformal_selected_90'].sum().mean())
    
    print(f"[+] Conformal Target Coverage   : 90.0%")
    print(f"[+] Empirical Realized Coverage : {empirical_coverage * 100:.2f}%")
    print(f"[+] Average Daily Set Size      : {avg_set_size:.1f} names")
    
    conformal_metrics = {
        "target_coverage": 0.90,
        "realized_coverage": round(empirical_coverage, 4),
        "avg_daily_selected_names": round(avg_set_size, 2),
        "sector_thresholds": thresholds
    }
    
    with open(os.path.join(out_dir, "metrics", "conformal_metrics.json"), "w") as f:
        json.dump(conformal_metrics, f, indent=4)
        
    # Overwrite shortlist with conformal columns
    df_conformal.to_parquet(oos_path, index=False)
    
    # Export human-readable CSV shortlist
    csv_shortlist_path = os.path.join(out_dir, "shortlists", "final_scored_shortlist.csv")
    latest_date = df_conformal['date'].max()
    latest_shortlist = df_conformal[df_conformal['date'] == latest_date].sort_values('percentile_rank', ascending=False)
    
    export_cols = ['date', 'symbol', 'sector', 'percentile_rank', 'calibrated_propensity', 'conformal_selected_90']
    latest_shortlist[export_cols].to_csv(csv_shortlist_path, index=False)
    print(f"[✓] Saved final latest shortlist CSV to {csv_shortlist_path}")
    
    return df_conformal

if __name__ == "__main__":
    run_conformal_pipeline()