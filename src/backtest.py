"""
src/backtest.py
Simulates Decile spread strategies, calculates turnover, applies realistic transaction
costs (15 bps), and computes Sharpe Ratio, Max Drawdown, and Monotonicity spread.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_decile_backtest(df: pd.DataFrame, transaction_cost_bps: float = 15.0) -> tuple:
    """
    Constructs 10 equal-weighted decile portfolios on each rebalance date.
    D10: Top conviction (highest predicted rank)
    D1 : Bottom conviction
    """
    df = df.copy()
    cost_drag = transaction_cost_bps / 10000.0  # 15 bps = 0.0015
    
    # Rebalance every 21 days
    unique_dates = np.sort(df['date'].unique())
    rebal_dates = unique_dates[::21]
    
    decile_returns = {f"D{d}": [] for d in range(1, 11)}
    decile_dates = []
    
    # Store constituents for turnover calculation
    prev_d10_holdings = set()
    d10_turnovers = []
    
    for dt in rebal_dates:
        cross_section = df[df['date'] == dt].dropna(subset=['ranker_score', 'fwd_ret_21d']).copy()
        
        if len(cross_section) < 10:
            continue
            
        try:
            cross_section['decile'] = pd.qcut(
                cross_section['ranker_score'].rank(method='first'),
                q=10,
                labels=[f"D{i}" for i in range(1, 11)]
            )
        except Exception:
            continue
            
        decile_dates.append(dt)
        
        # Calculate return per decile
        for d in range(1, 11):
            d_name = f"D{d}"
            sub_d = cross_section[cross_section['decile'] == d_name]
            # 21-day un-annualized holding return
            ret = sub_d['fwd_ret_21d'].mean()
            
            # Apply turnover cost to D10
            if d == 10:
                current_d10 = set(sub_d['symbol'])
                if prev_d10_holdings:
                    turnover = len(current_d10 - prev_d10_holdings) / len(current_d10)
                    ret -= (turnover * cost_drag)
                    d10_turnovers.append(turnover)
                prev_d10_holdings = current_d10
                
            decile_returns[d_name].append(ret)
            
    df_perf = pd.DataFrame(decile_returns, index=decile_dates)
    df_perf['D10_minus_D1'] = df_perf['D10'] - df_perf['D1']
    
    return df_perf, d10_turnovers

def calculate_portfolio_metrics(df_perf: pd.DataFrame, d10_turnovers: list) -> dict:
    """Computes Sharpe Ratio, CAGR, Max Drawdown, and Monotonicity score."""
    periods_per_year = 252 / 21 # ~12 monthly cycles
    
    d10_ret = df_perf['D10']
    spread_ret = df_perf['D10_minus_D1']
    
    # Annualized metrics
    ann_mean_d10 = d10_ret.mean() * periods_per_year
    ann_vol_d10 = d10_ret.std() * np.sqrt(periods_per_year)
    sharpe_d10 = ann_mean_d10 / (ann_vol_d10 if ann_vol_d10 > 1e-8 else 1.0)
    
    ann_spread_mean = spread_ret.mean() * periods_per_year
    ann_spread_vol = spread_ret.std() * np.sqrt(periods_per_year)
    sharpe_spread = ann_spread_mean / (ann_spread_vol if ann_spread_vol > 1e-8 else 1.0)
    
    # Cumulative Drawdown
    cum_d10 = (1.0 + d10_ret).cumprod()
    peak = cum_d10.cummax()
    drawdown = (cum_d10 - peak) / peak
    max_dd = float(drawdown.min())
    
    # Monotonicity check: correlation between Decile order (1-10) and mean return
    mean_deciles = [df_perf[f"D{i}"].mean() for i in range(1, 11)]
    monotonicity = float(np.corrcoef(np.arange(1, 11), mean_deciles)[0, 1])
    
    avg_turnover = float(np.mean(d10_turnovers)) if d10_turnovers else 0.40
    
    return {
        "d10_ann_return_pct": round(ann_mean_d10 * 100, 2),
        "d10_sharpe_ratio": round(sharpe_d10, 2),
        "d10_max_drawdown_pct": round(max_dd * 100, 2),
        "d10_d1_spread_ann_pct": round(ann_spread_mean * 100, 2),
        "spread_sharpe_ratio": round(sharpe_spread, 2),
        "decile_monotonicity": round(monotonicity, 3),
        "avg_rebalance_turnover_pct": round(avg_turnover * 100, 2)
    }

def run_backtest_pipeline(oos_path="outputs/shortlists/oos_ranked_predictions.parquet", out_dir="outputs"):
    """Runs backtest, outputs performance summary JSON and decile spread figure."""
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    
    print(f"[*] Loading predictions for backtest from {oos_path}...")
    df = pd.read_parquet(oos_path)
    
    print("[*] Running 21-day rebalanced decile simulation (15 bps cost)...")
    df_perf, d10_turnovers = run_decile_backtest(df, transaction_cost_bps=15.0)
    metrics = calculate_portfolio_metrics(df_perf, d10_turnovers)
    
    print(f"[+] D10 Net Annual Return   : {metrics['d10_ann_return_pct']}%")
    print(f"[+] D10 Net Sharpe Ratio    : {metrics['d10_sharpe_ratio']}")
    print(f"[+] D10 - D1 Spread Return  : {metrics['d10_d1_spread_ann_pct']}% (Sharpe: {metrics['spread_sharpe_ratio']})")
    print(f"[+] Decile Monotonicity     : {metrics['decile_monotonicity']}")
    print(f"[+] Average D10 Turnover    : {metrics['avg_rebalance_turnover_pct']}%")
    
    with open(os.path.join(out_dir, "metrics", "backtest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot Cumulative Decile Performance
    plt.figure(figsize=(10, 6))
    
    # Plot D1, D5, D10 and Spread
    d1_cum = (1.0 + df_perf['D1']).cumprod()
    d5_cum = (1.0 + df_perf['D5']).cumprod()
    d10_cum = (1.0 + df_perf['D10']).cumprod()
    spread_cum = (1.0 + df_perf['D10_minus_D1']).cumprod()
    
    plt.plot(df_perf.index, d10_cum, label=f"D10 Top Conviction (Net)", color="darkgreen", linewidth=2.5)
    plt.plot(df_perf.index, d5_cum, label="D5 Median Portfolio", color="gray", linestyle="--")
    plt.plot(df_perf.index, d1_cum, label="D1 Bottom Conviction", color="firebrick", linewidth=1.5)
    plt.plot(df_perf.index, spread_cum, label=f"D10 - D1 Long/Short Spread", color="dodgerblue", linewidth=2)
    
    plt.title("Decile Spread & Wealth Trajectory (21-Day Holding, 15 bps Cost)", fontsize=12, fontweight="bold")
    plt.xlabel("Rebalance Date")
    plt.ylabel("Cumulative Growth (Base = 1.0)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    fig_path = os.path.join(out_dir, "figures", "decile_spread_curve.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[✓] Decile spread curve saved to {fig_path}")

if __name__ == "__main__":
    run_backtest_pipeline()