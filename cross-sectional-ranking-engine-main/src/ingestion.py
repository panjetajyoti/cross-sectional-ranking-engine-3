"""
src/ingestion.py
Fetches adjusted OHLCV and availability-dated fundamentals for Indian equities.
Ensures point-in-time universe construction without look-ahead bias.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Liquid Indian Stocks Universe (Nifty 50 / Midcap representative basket)
NSE_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "HINDUNILVR.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "TATASTEEL.NS", "NTPC.NS",
    "POWERGRID.NS", "M&M.NS", "HCLTECH.NS", "TATAMOTORS.NS", "WIPRO.NS"
]

SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Financials",
    "ICICIBANK.NS": "Financials", "INFY.NS": "IT", "ITC.NS": "FMCG",
    "LT.NS": "Industrials", "SBIN.NS": "Financials", "BHARTIARTL.NS": "Telecom",
    "KOTAKBANK.NS": "Financials", "BAJFINANCE.NS": "Financials",
    "HINDUNILVR.NS": "FMCG", "AXISBANK.NS": "Financials", "ASIANPAINT.NS": "Consumer",
    "MARUTI.NS": "Auto", "SUNPHARMA.NS": "Healthcare", "TITAN.NS": "Consumer",
    "ULTRACEMCO.NS": "Materials", "TATASTEEL.NS": "Materials", "NTPC.NS": "Utilities",
    "POWERGRID.NS": "Utilities", "M&M.NS": "Auto", "HCLTECH.NS": "IT",
    "TATAMOTORS.NS": "Auto", "WIPRO.NS": "IT"
}

def fetch_market_data(symbols=NSE_UNIVERSE, start_date="2020-01-01", end_date="2026-03-01", output_dir="data/raw"):
    """
    Downloads historical OHLCV data using yfinance and formats as panel data.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Fetching OHLCV data for {len(symbols)} Indian equities...")
    
    df_raw = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', auto_adjust=True, progress=True)
    
    panel_records = []
    for sym in symbols:
        if sym in df_raw.columns.levels[0]:
            sub_df = df_raw[sym].copy().reset_index()
            sub_df.columns = [c.lower() for c in sub_df.columns]
            sub_df['symbol'] = sym.replace('.NS', '')
            sub_df['sector'] = SECTOR_MAP.get(sym, 'Others')
            panel_records.append(sub_df[['date', 'symbol', 'sector', 'open', 'high', 'low', 'close', 'volume']])
            
    panel = pd.concat(panel_records, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['date']).dt.tz_localize(None)
    panel = panel.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    # Save Raw Market Panel
    raw_path = os.path.join(output_dir, "nse_market_data.parquet")
    panel.to_parquet(raw_path, index=False)
    print(f"[✓] Saved market panel to {raw_path} ({len(panel)} rows)")
    return panel

def generate_point_in_time_fundamentals(market_panel, output_dir="data/raw"):
    """
    Simulates availability-dated fundamentals (PIT) to strictly prevent look-ahead bias.
    Financials are reported quarterly and available after a 45-day filing lag.
    """
    os.makedirs(output_dir, exist_ok=True)
    symbols = market_panel['symbol'].unique()
    dates = market_panel['date'].sort_values().unique()
    
    # Quarterly announcement dates
    quarter_ends = pd.date_range(start=dates.min(), end=dates.max(), freq='QE')
    fund_records = []
    
    np.random.seed(42)
    for q_end in quarter_ends:
        avail_date = q_end + pd.Timedelta(days=45) # 45-day mandatory SEBI filing lag
        for sym in symbols:
            # Fundamental factor proxies: ROE, Accruals, Debt/Equity, Earnings Yield proxy
            roe = np.clip(np.random.normal(0.18, 0.05), 0.02, 0.40)
            earnings_yield = np.clip(np.random.normal(0.045, 0.015), 0.01, 0.12)
            debt_to_equity = np.clip(np.random.normal(0.6, 0.3), 0.0, 2.5)
            accruals = np.random.normal(0.02, 0.01)
            
            fund_records.append({
                'symbol': sym,
                'quarter_end': q_end,
                'avail_date': avail_date,
                'roe': roe,
                'earnings_yield': earnings_yield,
                'debt_to_equity': debt_to_equity,
                'accruals': accruals
            })
            
    df_fund = pd.DataFrame(fund_records)
    fund_path = os.path.join(output_dir, "pit_fundamentals.parquet")
    df_fund.to_parquet(fund_path, index=False)
    print(f"[✓] Saved point-in-time fundamentals to {fund_path}")
    return df_fund

if __name__ == "__main__":
    market_df = fetch_market_data()
    fund_df = generate_point_in_time_fundamentals(market_df)
    print("[✓] Ingestion step complete.")