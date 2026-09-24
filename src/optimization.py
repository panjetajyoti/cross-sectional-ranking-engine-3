"""
src/optimization.py
Optuna-driven Bayesian Hyperparameter Optimisation targeting cross-sectional Rank IC
under Purged CV. Evaluates feature stability across temporal splits.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import scipy.stats as stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    'mom_12m_1m_neut', 'mom_6m_neut', 'mom_1m_neut',
    'vol_60d_neut', 'downside_vol_neut',
    'roe_neut', 'earnings_yield_neut', 'debt_to_equity_neut', 'accruals_neut'
]

def evaluate_fold_rank_ic(ranker, X_val, df_val):
    preds = ranker.predict(X_val)
    df_val = df_val.copy()
    df_val['score'] = preds
    
    daily_ics = []
    for _, grp in df_val.groupby('date'):
        valid = grp[['score', 'fwd_ret_21d']].dropna()
        if len(valid) >= 5:
            ic, _ = stats.spearmanr(valid['score'], valid['fwd_ret_21d'])
            if not np.isnan(ic):
                daily_ics.append(ic)
    return np.mean(daily_ics) if daily_ics else -1.0

def objective(trial, df, folds):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 31),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': 100,
        'random_state': 42,
        'verbose': -1
    }
    
    fold_ics = []
    for tr_dates, val_dates in folds:
        df_train = df[df['date'].isin(tr_dates)]
        df_val = df[df['date'].isin(val_dates)]
        
        train_groups = df_train.groupby('date', sort=False).size().values
        X_train, y_train = df_train[FEATURE_COLS], df_train['rank_relevance']
        X_val = df_val[FEATURE_COLS]
        
        ranker = lgb.LGBMRanker(**params)
        ranker.fit(X_train, y_train, group=train_groups)
        
        ic_val = evaluate_fold_rank_ic(ranker, X_val, df_val)
        fold_ics.append(ic_val)
        
    return np.mean(fold_ics)

def compute_feature_stability(df, folds, best_params, out_dir="outputs"):
    stability_records = []
    
    for fold_num, (tr_dates, _) in enumerate(folds, start=1):
        df_train = df[df['date'].isin(tr_dates)]
        train_groups = df_train.groupby('date', sort=False).size().values
        
        ranker = lgb.LGBMRanker(**best_params)
        ranker.fit(df_train[FEATURE_COLS], df_train['rank_relevance'], group=train_groups)
        
        gains = ranker.booster_.feature_importance(importance_type='gain')
        for feat, g in zip(FEATURE_COLS, gains):
            stability_records.append({'fold': fold_num, 'feature': feat, 'gain': g})
            
    df_stab = pd.DataFrame(stability_records)
    summary = df_stab.groupby('feature')['gain'].agg(['mean', 'std']).reset_index()
    summary['stability_score'] = summary['mean'] / (summary['std'] + 1e-6)
    summary = summary.sort_values('mean', ascending=True)
    
    # Feature Importance Plot
    plt.figure(figsize=(9, 5))
    plt.barh(summary['feature'], summary['mean'], xerr=summary['std'], color='teal', alpha=0.75, capsize=4)
    plt.title("Feature Stability & Importance (Mean Gain across Folds)", fontsize=11, fontweight="bold")
    plt.xlabel("Average Gain")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(out_dir, "figures", "feature_importance_stability.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Feature stability plot saved to {plot_path}")
    
    return summary.to_dict(orient='records')

def run_optimization_pipeline(data_path="data/curated_pit/labeled_dataset.parquet", out_dir="outputs", n_trials=10):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    
    print(f"[*] Reading dataset from {data_path}...")
    df = pd.read_parquet(data_path).sort_values('date').reset_index(drop=True)
    
    unique_dates = np.sort(df['date'].unique())
    n = len(unique_dates)
    fold_size = n // 4
    folds = [
        (unique_dates[:fold_size * 2], unique_dates[fold_size * 2 + 21:fold_size * 3]),
        (unique_dates[:fold_size * 3], unique_dates[fold_size * 3 + 21:])
    ]
    
    print(f"[*] Launching Optuna HPO ({n_trials} trials, target: maximize Validation Rank IC)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df, folds), n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': 120,
        'random_state': 42,
        'verbose': -1
    })
    
    print(f"[+] Best Validation Rank IC: {study.best_value:.4f}")
    print(f"[+] Optimal Hyperparameters : {best_params}")
    
    print("[*] Computing cross-fold feature stability...")
    stability_data = compute_feature_stability(df, folds, best_params, out_dir=out_dir)
    
    opt_results = {
        "best_rank_ic": round(float(study.best_value), 4),
        "best_params": study.best_params,
        "feature_stability": stability_data
    }
    
    out_json = os.path.join(out_dir, "metrics", "hpo_and_feature_stability.json")
    with open(out_json, "w") as f:
        json.dump(opt_results, f, indent=4)
        
    print(f"[✓] Optimization and stability metrics saved to {out_json}")
    return opt_results

if __name__ == "__main__":
    run_optimization_pipeline()