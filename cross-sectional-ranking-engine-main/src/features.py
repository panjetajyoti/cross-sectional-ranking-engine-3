"""
src/features.py
Constructs 6 factor families, merges Point-in-Time data, performs cross-sectional
winsorisation and sector/size neutralisation. Saves curated dataset to data/curated_pit/.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Momentum, Volatility, and Liquidity factors per stock."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Daily Returns
    df['ret_1d'] = df.groupby('symbol')['close'].pct_change()
    
    # 1. Momentum Family
    # 12M - 1M momentum (skip last 21 trading days to avoid short-term reversal)
    df['mom_12m_1m'] = df.groupby('symbol')['close'].transform(
        lambda x: (x.shift(21) / x.shift(252)) - 1.0
    )
    df['mom_6m'] = df.groupby('symbol')['close'].transform(
        lambda x: (x / x.shift(126)) - 1.0
    )
    df['mom_1m'] = df.groupby('symbol')['close'].transform(
        lambda x: (x / x.shift(21)) - 1.0
    )
    
    # 2. Volatility / Low-Risk Family
    df['vol_60d'] = df.groupby('symbol')['ret_1d'].transform(
        lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252)
    )
    df['downside_vol'] = df.groupby('symbol')['ret_1d'].transform(
        lambda x: x.rolling(60, min_periods=30).apply(
            lambda r: np.std(r[r < 0]) * np.sqrt(252) if (r < 0).sum() > 2 else 0.0,
            raw=True
        )
    )
    
    # 3. Size & Liquidity/Flow Family
    # Rupee Turnover proxy
    df['turnover_val'] = df['close'] * df['volume']
    df['adv_20d'] = df.groupby('symbol')['turnover_val'].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    df['log_size_proxy'] = np.log(df['adv_20d'].clip(lower=1.0))
    
    return df

def merge_pit_fundamentals(market_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Point-in-Time join: fundamental filings become available strictly ON or AFTER avail_date.
    Completely eliminates forward-looking look-ahead leakage.
    """
    market_df['date'] = pd.to_datetime(market_df['date'])
    fund_df['avail_date'] = pd.to_datetime(fund_df['avail_date'])
    
    merged_list = []
    symbols = market_df['symbol'].unique()
    
    fund_cols = ['roe', 'earnings_yield', 'debt_to_equity', 'accruals']
    
    for sym in symbols:
        m_sub = market_df[market_df['symbol'] == sym].sort_values('date').copy()
        f_sub = fund_df[fund_df['symbol'] == sym].sort_values('avail_date').copy()
        
        if f_sub.empty:
            for c in fund_cols:
                m_sub[c] = np.nan
            merged_list.append(m_sub)
            continue
            
        merged = pd.merge_asof(
            m_sub,
            f_sub[['avail_date'] + fund_cols],
            left_on='date',
            right_on='avail_date',
            direction='backward'
        )
        merged_list.append(merged)
        
    final_df = pd.concat(merged_list, ignore_index=True)
    return final_df.sort_values(['date', 'symbol']).reset_index(drop=True)

def winsorise_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    """Clips extreme outliers at percentile thresholds."""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return s
    q_low = s_clean.quantile(lower)
    q_high = s_clean.quantile(upper)
    return s.clip(lower=q_low, upper=q_high)

def neutralise_cross_section(df_panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Groups by trading date, then regresses each raw factor cross-sectionally
    on Sector dummies and Log-Size proxy. Saves residuals as neutralised factors.
    """
    results = []
    grouped = df_panel.groupby('date')
    
    for dt, group in grouped:
        group = group.copy()
        n_names = len(group)
        
        # Cross-sectional design matrix
        sector_dummies = pd.get_dummies(group['sector'], drop_first=True, dtype=float)
        size_cov = group[['log_size_proxy']].fillna(group['log_size_proxy'].median())
        
        X = pd.concat([sector_dummies, size_cov], axis=1)
        X = sm.add_constant(X)
        
        for col in feature_cols:
            y = group[col].copy()
            y = winsorise_series(y)
            
            valid_mask = ~(y.isna() | X.isna().any(axis=1))
            
            if valid_mask.sum() >= 6:
                try:
                    ols = sm.OLS(y[valid_mask], X[valid_mask]).fit()
                    resid = pd.Series(index=y.index, dtype=float)
                    resid.loc[valid_mask] = ols.resid
                    
                    std_val = resid.std()
                    group[f"{col}_neut"] = (resid - resid.mean()) / (std_val if std_val > 1e-8 else 1.0)
                except Exception:
                    std_val = y.std()
                    group[f"{col}_neut"] = (y - y.mean()) / (std_val if std_val > 1e-8 else 1.0)
            else:
                std_val = y.std()
                group[f"{col}_neut"] = (y - y.mean()) / (std_val if std_val > 1e-8 else 1.0)
                
        results.append(group)
        
    return pd.concat(results, ignore_index=True)

def build_feature_pipeline(raw_dir="data/raw", out_dir="data/curated_pit"):
    """End-to-end execution of Point-in-Time feature engineering."""
    os.makedirs(out_dir, exist_ok=True)
    print("[*] Loading raw market and fundamental parquet files...")
    
    market_df = pd.read_parquet(os.path.join(raw_dir, "nse_market_data.parquet"))
    fund_df = pd.read_parquet(os.path.join(raw_dir, "pit_fundamentals.parquet"))
    
    print("[*] Computing price and volume factors...")
    market_feat = compute_technical_factors(market_df)
    
    print("[*] Performing Point-in-Time backward as-of merge...")
    pit_df = merge_pit_fundamentals(market_feat, fund_df)
    
    # 6 Feature families to neutralise
    feature_cols = [
        'mom_12m_1m', 'mom_6m', 'mom_1m',
        'vol_60d', 'downside_vol',
        'roe', 'earnings_yield', 'debt_to_equity', 'accruals'
    ]
    
    # Filter rows with sufficient price history
    pit_df = pit_df.dropna(subset=['mom_12m_1m']).copy()
    
    print(f"[*] Running cross-sectional winsorisation and sector/size residualisation across {pit_df['date'].nunique()} trading dates...")
    curated_df = neutralise_cross_section(pit_df, feature_cols)
    
    out_path = os.path.join(out_dir, "curated_features_pit.parquet")
    curated_df.to_parquet(out_path, index=False)
    print(f"[✓] Curated PIT features saved to {out_path} ({len(curated_df)} rows)")
    return curated_df

if __name__ == "__main__":
    build_feature_pipeline()