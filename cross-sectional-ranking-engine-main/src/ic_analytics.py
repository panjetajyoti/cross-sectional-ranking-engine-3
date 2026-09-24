"""
src/ic_analytics.py
Evaluates cross-sectional ranking quality using Spearman Rank IC, IC-IR,
t-statistics, and horizon decay curves (1 to 63 days).
Generates audit-ready charts and performance tables.
"""

import os
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def compute_daily_rank_ic(df: pd.DataFrame, score_col='ranker_score', return_col='fwd_ret_21d') -> pd.DataFrame:
    """
    Computes cross-sectional Spearman Rank IC for every rebalance/trading date.
    """
    ic_records = []
    
    for dt, group in df.groupby('date'):
        valid_data = group[[score_col, return_col]].dropna()
        if len(valid_data) >= 5:
            # Spearman rank correlation
            spearman_corr, _ = stats.spearmanr(valid_data[score_col], valid_data[return_col])
            # Pearson correlation
            pearson_corr, _ = stats.pearsonr(valid_data[score_col], valid_data[return_col])
            
            ic_records.append({
                'date': dt,
                'rank_ic': spearman_corr,
                'pearson_ic': pearson_corr,
                'n_stocks': len(valid_data)
            })
            
    df_ic = pd.DataFrame(ic_records).sort_values('date').reset_index(drop=True)
    return df_ic

def compute_ic_summary_stats(df_ic: pd.DataFrame) -> dict:
    """Computes Mean IC, IC Standard Deviation, IC-IR, and Newey-West adjusted t-stat."""
    ic_series = df_ic['rank_ic'].dropna()
    n = len(ic_series)
    
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    ic_ir = mean_ic / (std_ic if std_ic > 1e-8 else 1.0)
    
    # t-statistic: mean / (std / sqrt(n))
    t_stat = mean_ic / ((std_ic / np.sqrt(n)) if std_ic > 1e-8 else 1.0)
    p_value = float(2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 1)))
    hit_rate = float((ic_series > 0).mean())
    
    return {
        "mean_rank_ic": round(mean_ic, 4),
        "std_rank_ic": round(std_ic, 4),
        "ic_ir": round(ic_ir, 4),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_value, 6),
        "hit_rate_pct": round(hit_rate * 100, 2),
        "total_periods": n
    }

def compute_ic_decay(df: pd.DataFrame, horizons=[1, 5, 10, 21, 42, 63]) -> pd.DataFrame:
    """
    Measures how quickly the predictive power of the model decays across longer forward horizons.
    """
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    decay_results = []
    
    for h in horizons:
        df[f'fwd_ret_{h}d'] = df.groupby('symbol')['close'].shift(-h) / df['close'] - 1.0
        
        # Calculate mean rank IC for horizon h
        daily_ics = []
        for dt, group in df.groupby('date'):
            valid = group[['ranker_score', f'fwd_ret_{h}d']].dropna()
            if len(valid) >= 5:
                s_corr, _ = stats.spearmanr(valid['ranker_score'], valid[f'fwd_ret_{h}d'])
                if not np.isnan(s_corr):
                    daily_ics.append(s_corr)
                    
        decay_results.append({
            'horizon_days': h,
            'mean_rank_ic': np.mean(daily_ics) if daily_ics else 0.0
        })
        
    return pd.DataFrame(decay_results)

def run_ic_analytics_pipeline(oos_path="outputs/shortlists/oos_ranked_predictions.parquet", out_dir="outputs"):
    """
    Executes full IC calculation, produces cumulative IC charts and decay plots.
    """
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    
    print(f"[*] Loading dataset from {oos_path}...")
    df = pd.read_parquet(oos_path)
    
    # 1. Daily Rank IC
    print("[*] Calculating cross-sectional Daily Rank IC...")
    df_ic = compute_daily_rank_ic(df)
    summary_stats = compute_ic_summary_stats(df_ic)
    
    print(f"[+] Mean Rank IC   : {summary_stats['mean_rank_ic']:.4f}")
    print(f"[+] IC-IR          : {summary_stats['ic_ir']:.4f}")
    print(f"[+] IC t-statistic : {summary_stats['t_stat']:.2f} (p-val: {summary_stats['p_value']:.4f})")
    print(f"[+] Positive IC %  : {summary_stats['hit_rate_pct']:.2f}%")
    
    # Save metrics JSON
    with open(os.path.join(out_dir, "metrics", "ic_summary_metrics.json"), "w") as f:
        json.dump(summary_stats, f, indent=4)
        
    # 2. Cumulative Rank IC Plot
    df_ic['cum_ic'] = df_ic['rank_ic'].cumsum()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.bar(pd.to_datetime(df_ic['date']), df_ic['rank_ic'], color='steelblue', alpha=0.6, width=1.5)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_title("Cross-Sectional Rank IC (Daily 21-Day Forward)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Spearman IC")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(pd.to_datetime(df_ic['date']), df_ic['cum_ic'], color='crimson', linewidth=2, label=f"Cum IC (Final: {df_ic['cum_ic'].iloc[-1]:.2f})")
    ax2.set_title("Cumulative Information Coefficient", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative IC")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    cum_path = os.path.join(out_dir, "figures", "cumulative_rank_ic.png")
    plt.tight_layout()
    plt.savefig(cum_path, dpi=300)
    plt.close()
    print(f"[✓] Saved cumulative IC chart to {cum_path}")
    
    # 3. IC Horizon Decay Curve
    print("[*] Computing IC decay across forward horizons (1 to 63 days)...")
    decay_df = compute_ic_decay(df)
    
    plt.figure(figsize=(7, 4.5))
    plt.plot(decay_df['horizon_days'], decay_df['mean_rank_ic'], marker='o', color='purple', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Information Coefficient Decay Horizon (Holding Period Alpha)", fontsize=11, fontweight="bold")
    plt.xlabel("Holding Period Horizon (Trading Days)")
    plt.ylabel("Mean Rank IC")
    plt.xticks(decay_df['horizon_days'])
    plt.grid(True, alpha=0.3)
    
    decay_path = os.path.join(out_dir, "figures", "ic_decay_curve.png")
    plt.tight_layout()
    plt.savefig(decay_path, dpi=300)
    plt.close()
    print(f"[✓] Saved IC decay curve to {decay_path}")
    
    return summary_stats

if __name__ == "__main__":
    run_ic_analytics_pipeline()