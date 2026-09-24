"""
src/labeling.py
Constructs 21-day forward relative return targets and rank relevance bins.
Prepares train/validation ready cross-sectional panel data.
"""

import os
import pandas as pd
import numpy as np

def compute_forward_labels(df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    Computes H-day forward return per stock, cross-sectional excess return,
    binary outperformance label, and quintile relevance grade for LambdaMART.
    """
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 1. Forward 21-day price return
    df['fwd_close'] = df.groupby('symbol')['close'].shift(-horizon)
    df['fwd_ret_21d'] = (df['fwd_close'] / df['close']) - 1.0
    
    # Drop rows at the end where forward return is unknown
    labeled_df = df.dropna(subset=['fwd_ret_21d']).copy()
    
    # 2. Cross-Sectional Relative Target (Date-wise)
    def label_cross_section(g):
        med_ret = g['fwd_ret_21d'].median()
        # Relative return above median
        g['rel_ret_21d'] = g['fwd_ret_21d'] - med_ret
        # Binary target: 1 if outperformed cross-sectional median, else 0
        g['binary_outperform'] = (g['rel_ret_21d'] > 0).astype(int)
        
        # Discrete relevance grades 0 to 4 (Quintiles) for Learning-to-Rank (LambdaMART)
        try:
            g['rank_relevance'] = pd.qcut(g['fwd_ret_21d'], q=5, labels=[0, 1, 2, 3, 4]).astype(int)
        except Exception:
            # Fallback if ties exist
            ranks = g['fwd_ret_21d'].rank(pct=True)
            g['rank_relevance'] = pd.cut(ranks, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0, 1, 2, 3, 4], include_lowest=True).astype(int)
            
        return g

    print(f"[*] Calculating 21-day forward relative labels across {labeled_df['date'].nunique()} trading dates...")
    labeled_df = labeled_df.groupby('date', group_keys=False).apply(label_cross_section)
    return labeled_df

def run_labeling_pipeline(curated_path="data/curated_pit/curated_features_pit.parquet", out_dir="data/curated_pit"):
    """Reads curated PIT features, generates labels, and saves the final modeling dataset."""
    print(f"[*] Reading features from {curated_path}...")
    df_features = pd.read_parquet(curated_path)
    
    df_labeled = compute_forward_labels(df_features, horizon=21)
    
    out_file = os.path.join(out_dir, "labeled_dataset.parquet")
    df_labeled.to_parquet(out_file, index=False)
    print(f"[✓] Labeled dataset successfully saved to {out_file} ({len(df_labeled)} rows)")
    return df_labeled

if __name__ == "__main__":
    run_labeling_pipeline()