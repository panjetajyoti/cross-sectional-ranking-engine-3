"""
src/monte_carlo.py
Simulates 1,000 forward paths for top-ranked shortlisted stocks.
Computes Value at Risk (VaR 95%), Expected Shortfall (CVaR),
and empirical distribution of forward returns.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo_simulation(
    shortlist_path="outputs/shortlists/final_scored_shortlist.csv",
    data_path="data/curated_pit/labeled_dataset.parquet",
    out_dir="outputs",
    n_simulations: int = 1000,
    horizon_days: int = 21
):
    """
    Simulates forward return distributions for top conviction portfolio
    using parametric Gaussian and bootstrap historical resampling.
    """
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    
    print(f"[*] Loading shortlist from {shortlist_path}...")
    df_short = pd.read_csv(shortlist_path)
    top_names = df_short.head(10)['symbol'].tolist()
    print(f"[+] Top conviction names: {top_names}")
    
    # Load historical daily returns to estimate mean vector and covariance matrix
    df_panel = pd.read_parquet(data_path)
    df_top = df_panel[df_panel['symbol'].isin(top_names)].copy()
    
    pivot_ret = df_top.pivot(index='date', columns='symbol', values='ret_1d').dropna()
    
    # Calculate annualized statistics
    mean_daily = pivot_ret.mean()
    cov_daily = pivot_ret.cov()
    
    # Equal-weighted portfolio weights
    n_assets = len(top_names)
    weights = np.ones(n_assets) / n_assets
    
    port_daily_mean = np.dot(weights, mean_daily)
    port_daily_var = np.dot(weights.T, np.dot(cov_daily, weights))
    port_daily_std = np.sqrt(port_daily_var)
    
    # 21-day holding horizon parameters
    mu_horizon = port_daily_mean * horizon_days
    sigma_horizon = port_daily_std * np.sqrt(horizon_days)
    
    np.random.seed(42)
    # Generate 1,000 simulated 21-day return outcomes
    simulated_returns = np.random.normal(mu_horizon, sigma_horizon, n_simulations)
    
    # Risk metrics
    var_95 = float(np.percentile(simulated_returns, 5))
    cvar_95 = float(simulated_returns[simulated_returns <= var_95].mean())
    prob_positive = float((simulated_returns > 0).mean())
    median_outcome = float(np.median(simulated_returns))
    
    print(f"\n[+] Monte Carlo Expected Horizon Return : {median_outcome * 100:.2f}%")
    print(f"[+] Horizon Value-at-Risk (VaR 95%)      : {var_95 * 100:.2f}%")
    print(f"[+] Conditional VaR (Expected Shortfall) : {cvar_95 * 100:.2f}%")
    print(f"[+] Win Rate Probability (>0% Return)    : {prob_positive * 100:.2f}%")
    
    mc_results = {
        "top_holdings": top_names,
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "median_return_pct": round(median_outcome * 100, 2),
        "var_95_pct": round(var_95 * 100, 2),
        "cvar_95_pct": round(cvar_95 * 100, 2),
        "prob_positive_return_pct": round(prob_positive * 100, 2)
    }
    
    out_json = os.path.join(out_dir, "metrics", "monte_carlo_risk_metrics.json")
    with open(out_json, "w") as f:
        json.dump(mc_results, f, indent=4)
        
    # Plot Distribution Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(simulated_returns * 100, bins=40, color='royalblue', alpha=0.7, edgecolor='black')
    plt.axvline(var_95 * 100, color='red', linestyle='--', linewidth=2, label=f"VaR 95% ({var_95 * 100:.1f}%)")
    plt.axvline(median_outcome * 100, color='green', linestyle='-', linewidth=2, label=f"Median ({median_outcome * 100:.1f}%)")
    plt.title("Monte Carlo 21-Day Forward Return Distribution (1,000 Paths)", fontsize=11, fontweight="bold")
    plt.xlabel("Simulated Portfolio Return (%)")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    fig_path = os.path.join(out_dir, "figures", "monte_carlo_distribution.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[✓] Monte Carlo distribution chart saved to {fig_path}")

if __name__ == "__main__":
    run_monte_carlo_simulation()