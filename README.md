# Cross-Sectional Ranking & Propensity Engine for Indian Equities
**Institutional Research Project | Zetheta Algorithms**  
**Corporate Identity Number (CIN)**: U62012MH2023PTC410415  

## Overview
A production-grade, single-engine quantitative workflow that scores Indian equities by their probability of outperforming peers over a forward 21-day holding horizon. Implements LightGBM LambdaMART under Purged & Embargoed Cross-Validation, Platt/Isotonic Probability Calibration, Mondrian Conformal Prediction wrappers, and cryptographic SHA-256 audit logging.

## Core Features
- **Point-in-Time Data Pipeline**: Backward as-of joins with a 45-day SEBI filing lag constraint to eliminate look-ahead bias.
- **Factor Engineering**: 6 factor families neutralised cross-sectionally against Sector dummies and Log-Size.
- **LambdaMART Ranker**: NDCG@10 optimization under temporal folds with a 21-day embargo buffer.
- **Dual Utility**: Ordinal conviction rank combined with calibrated out-performance probabilities.
- **Evaluation**: Daily Spearman Rank IC, IC-IR, t-statistic, and horizon decay curves (1–63 days).
- **Decile Backtest**: Long/short spread simulation net of 15 bps slippage and turnover penalties.
- **Agentic Governance**: Automated Model Card generation and cryptographic hash chaining.

## Quick Start
```powershell
# Setup environment
pip install -r requirements.txt

# Run complete pipeline
python src/ingestion.py
python src/features.py
python src/labeling.py
python src/ranker_lambdamart.py
python src/calibration.py
python src/ic_analytics.py
python src/conformal.py
python src/backtest.py
python src/optimization.py
python src/ensembling.py
python src/monte_carlo.py

# Run autonomous agent & compliance check
python -m agents_aws.agent_workflow