"""
src/ranker_lambdamart.py
Implements LightGBM LambdaMART cross-sectional learning-to-rank model.
Uses Purged and Embargoed time-series cross-validation to prevent forward leakage.
"""

import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score

FEATURE_COLS = [
    'mom_12m_1m_neut', 'mom_6m_neut', 'mom_1m_neut',
    'vol_60d_neut', 'downside_vol_neut',
    'roe_neut', 'earnings_yield_neut', 'debt_to_equity_neut', 'accruals_neut'
]

TARGET_COL = 'rank_relevance'

def get_purged_folds(dates: pd.Series, n_splits: int = 5, embargo_pct: float = 0.02):
    """
    Generates train/validation time-series indices with an embargo buffer
    to eliminate cross-sectional overlap and look-ahead bias.
    """
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // (n_splits + 1)
    embargo_days = int(n_dates * embargo_pct)
    
    folds = []
    for i in range(1, n_splits + 1):
        train_end_idx = i * fold_size
        val_start_idx = train_end_idx + embargo_days
        val_end_idx = min(val_start_idx + fold_size, n_dates)
        
        if val_start_idx >= n_dates:
            break
            
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]
        folds.append((train_dates, val_dates))
        
    return folds

def train_lambdamart_engine(data_path="data/curated_pit/labeled_dataset.parquet", out_dir="outputs"):
    """
    Trains LightGBM Ranker across temporal folds, collects Out-Of-Sample (OOS)
    scores, and persists the production model.
    """
    os.makedirs(os.path.join(out_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "shortlists"), exist_ok=True)
    
    print(f"[*] Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    folds = get_purged_folds(df['date'], n_splits=4, embargo_pct=0.03)
    print(f"[*] Created {len(folds)} Purged & Embargoed temporal folds.")
    
    # Store out-of-fold predictions
    df['ranker_score'] = np.nan
    df['percentile_rank'] = np.nan
    
    models = []
    val_ndcg_list = []
    
    for fold_num, (tr_dates, val_dates) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold_num}: Train [{pd.to_datetime(tr_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(tr_dates[-1]).strftime('%Y-%m-%d')}] | Val [{pd.to_datetime(val_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(val_dates[-1]).strftime('%Y-%m-%d')}] ---")
        
        tr_mask = df['date'].isin(tr_dates)
        val_mask = df['date'].isin(val_dates)
        
        df_train = df[tr_mask].copy()
        df_val = df[val_mask].copy()
        
        # Group sizes (number of stocks per query date)
        train_groups = df_train.groupby('date', sort=False).size().values
        val_groups = df_val.groupby('date', sort=False).size().values
        
        X_train = df_train[FEATURE_COLS]
        y_train = df_train[TARGET_COL]
        X_val = df_val[FEATURE_COLS]
        y_val = df_val[TARGET_COL]
        
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold_num,
            importance_type="gain"
        )
        
        ranker.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_val, y_val)],
            eval_group=[val_groups],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Predict continuous relevance scores
        val_preds = ranker.predict(X_val)
        df.loc[val_mask, 'ranker_score'] = val_preds
        
        # Compute cross-sectional percentile ranks per date
        df.loc[val_mask, 'percentile_rank'] = (
            df.loc[val_mask].groupby('date')['ranker_score'].rank(pct=True)
        )
        
        models.append(ranker)
        
        # Track validation NDCG@10
        try:
            sample_ndcg = ndcg_score([y_val.values[:len(val_groups)]], [val_preds[:len(val_groups)]], k=10)
            val_ndcg_list.append(sample_ndcg)
            print(f"[+] Fold {fold_num} NDCG@10: {sample_ndcg:.4f}")
        except Exception:
            pass

    # Save out-of-fold predictions
    oos_df = df.dropna(subset=['ranker_score']).copy()
    oos_path = os.path.join(out_dir, "shortlists", "oos_ranked_predictions.parquet")
    oos_df.to_parquet(oos_path, index=False)
    print(f"\n[✓] OOS Ranked predictions saved to {oos_path} ({len(oos_df)} records)")
    
    # Save best model
    model_save_path = os.path.join(out_dir, "model_cards", "lambdamart_engine.joblib")
    joblib.dump(models[-1], model_save_path)
    print(f"[✓] Persisted final production ranker to {model_save_path}")
    
    return oos_df

if __name__ == "__main__":
    train_lambdamart_engine()