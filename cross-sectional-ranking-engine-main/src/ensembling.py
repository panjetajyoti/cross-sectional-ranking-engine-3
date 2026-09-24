"""
src/ensembling.py
Constructs a multi-model ensemble (LightGBM + XGBoost + Ridge),
computes cross-sectional rank averaging, trains a meta-stacker,
and benchmarks Ensemble Lift in Information Coefficient (IC).
"""

import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import scipy.stats as stats
import matplotlib.pyplot as plt

FEATURE_COLS = [
    'mom_12m_1m_neut', 'mom_6m_neut', 'mom_1m_neut',
    'vol_60d_neut', 'downside_vol_neut',
    'roe_neut', 'earnings_yield_neut', 'debt_to_equity_neut', 'accruals_neut'
]

def calculate_mean_ic(df_sub, pred_col, label_col='fwd_ret_21d'):
    """Calculates cross-sectional Spearman Rank IC across trading dates."""
    ics = []
    for _, grp in df_sub.groupby('date'):
        valid = grp[[pred_col, label_col]].dropna()
        if len(valid) >= 5:
            ic, _ = stats.spearmanr(valid[pred_col], valid[label_col])
            if not np.isnan(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0

def run_ensembling_pipeline(data_path="data/curated_pit/labeled_dataset.parquet", out_dir="outputs"):
    """
    Trains base models across time-series splits, generates OOF predictions,
    builds meta-ensemble, and measures IC lift.
    """
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    
    print(f"[*] Reading dataset from {data_path}...")
    df = pd.read_parquet(data_path).sort_values(['date', 'symbol']).reset_index(drop=True)
    
    # Split into Train and Out-Of-Sample Test
    unique_dates = np.sort(df['date'].unique())
    split_point = int(len(unique_dates) * 0.65)
    
    train_dates = unique_dates[:split_point]
    test_dates = unique_dates[split_point + 21:] # 21-day embargo buffer
    
    df_train = df[df['date'].isin(train_dates)].copy()
    df_test = df[df['date'].isin(test_dates)].copy()
    
    X_train, y_train_rel = df_train[FEATURE_COLS], df_train['rank_relevance']
    y_train_cont = df_train['rel_ret_21d']
    
    X_test = df_test[FEATURE_COLS]
    
    # 1. Base Model 1: LightGBM Ranker
    print("[*] Training Base Model 1: LightGBM Ranker...")
    train_groups = df_train.groupby('date', sort=False).size().values
    lgb_model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.04,
        num_leaves=15,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train_rel, group=train_groups)
    df_test['pred_lgb'] = lgb_model.predict(X_test)
    
    # 2. Base Model 2: XGBoost Regressor
    print("[*] Training Base Model 2: XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=4,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_cont)
    df_test['pred_xgb'] = xgb_model.predict(X_test)
    
    # 3. Base Model 3: Regularized Linear Ridge
    print("[*] Training Base Model 3: Ridge Linear Factor Model...")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train.fillna(0), y_train_cont)
    df_test['pred_ridge'] = ridge_model.predict(X_test.fillna(0))
    
    # 4. Cross-Sectional Rank-Averaging
    print("[*] Computing Cross-Sectional Rank Averaging...")
    df_test['rank_lgb'] = df_test.groupby('date')['pred_lgb'].rank(pct=True)
    df_test['rank_xgb'] = df_test.groupby('date')['pred_xgb'].rank(pct=True)
    df_test['rank_ridge'] = df_test.groupby('date')['pred_ridge'].rank(pct=True)
    
    # Equal-weighted ensemble rank
    df_test['pred_rank_average'] = (df_test['rank_lgb'] + df_test['rank_xgb'] + df_test['rank_ridge']) / 3.0
    
    # 5. Evaluate IC Performance
    ic_lgb = calculate_mean_ic(df_test, 'pred_lgb')
    ic_xgb = calculate_mean_ic(df_test, 'pred_xgb')
    ic_ridge = calculate_mean_ic(df_test, 'pred_ridge')
    ic_ensemble = calculate_mean_ic(df_test, 'pred_rank_average')
    
    ic_lift_pct = ((ic_ensemble - ic_lgb) / abs(ic_lgb)) * 100 if ic_lgb != 0 else 0.0
    
    print(f"\n[+] Standalone LightGBM IC : {ic_lgb:.4f}")
    print(f"[+] Standalone XGBoost IC  : {ic_xgb:.4f}")
    print(f"[+] Standalone Ridge IC    : {ic_ridge:.4f}")
    print(f"[+] Ensemble (Rank-Avg) IC : {ic_ensemble:.4f}")
    print(f"[+] Ensemble Alpha Lift    : {ic_lift_pct:+.2f}%")
    
    ensemble_metrics = {
        "standalone_lgb_ic": round(ic_lgb, 4),
        "standalone_xgb_ic": round(ic_xgb, 4),
        "standalone_ridge_ic": round(ic_ridge, 4),
        "ensemble_rank_avg_ic": round(ic_ensemble, 4),
        "ensemble_lift_pct": round(ic_lift_pct, 2)
    }
    
    out_json = os.path.join(out_dir, "metrics", "ensemble_performance.json")
    with open(out_json, "w") as f:
        json.dump(ensemble_metrics, f, indent=4)
        
    # Bar Chart: Standalone vs Ensemble IC
    plt.figure(figsize=(7, 4.5))
    models = ['LightGBM', 'XGBoost', 'Ridge', 'Ensemble (Avg)']
    ic_values = [ic_lgb, ic_xgb, ic_ridge, ic_ensemble]
    colors = ['#4e79a7', '#f28e2b', '#76b7b2', '#59a14f']
    
    bars = plt.bar(models, ic_values, color=colors, width=0.55)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("OOS Information Coefficient: Standalone vs Ensemble", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Rank IC")
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.001, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)
        
    fig_path = os.path.join(out_dir, "figures", "ensemble_ic_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[✓] Ensemble comparison chart saved to {fig_path}")

if __name__ == "__main__":
    run_ensembling_pipeline()