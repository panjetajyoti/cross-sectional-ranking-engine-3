# INSTITUTIONAL PITCH DECK: CROSS-SECTIONAL RANKING & PROPENSITY ENGINE
**Role**: Financial Data Analyst / Quant Desk  
**Entity**: Zetheta Algorithms Private Limited (CIN: U62012MH2023PTC410415)  
**Target Audience**: Head of Research & Investment Committee (AMC)  

---

### Slide 1: Title & Corporate Identification
- **Title**: Unified Cross-Sectional Ranking & Propensity Engine for Indian Equities
- **Subtitle**: Direction over Magnitude: Transitioning Indian AMCs from Point Forecasts to Calibrated Relative Ordering
- **Metadata**: Desk Rotation Alpha Foundry | CIN: U62012MH2023PTC410415

### Slide 2: The Core Industry Crisis
- Three decades of Indian buy-side quant failed due to absolute price extrapolation (e.g., forecasting 'Stock X +12%').
- Idiosyncratic noise dominates single-stock levels; relative ordering alone survives out-of-sample.
- SIP-driven AUM expansion (>₹70 Lakh Cr in 2025) compresses alpha and demands audit-defensible selection workflows.

### Slide 3: Executive Mandate & Single-Engine Architecture
- Mandate: Replace point predictors with a single calibrated learning-to-rank engine.
- Dual Utility: Single score serves simultaneously as an ordinal conviction rank and an isotonic/Platt calibrated win probability P(Outperform).
- Verification Layer: Statistically certified via Spearman Information Coefficient (IC) and Mondrian conformal wrappers.

### Slide 4: Point-in-Time Data Engineering & Leakage Defense
- Look-ahead bias elimination: SEBI quarterly financial filings matched strictly with a 45-day mandatory filing lag via backward as-of merges.
- Survivorship bias defense: Dynamic equity universe with corporate action adjustment across splits, dividends, and rights.

### Slide 5: The 6 Factor Families
- Momentum (12M-1M skip-month, 6M, 1M momentum).
- Volatility & Downside Risk (60D annualized vol, semi-variance downside volatility).
- Quality (Point-in-Time ROE, Accruals anomaly).
- Value (Earnings Yield proxy).
- Size & Liquidity (20-day average rupee turnover, log market-cap proxy).
- Flow / Reversal Dynamics.

### Slide 6: Cross-Sectional Winsorisation & Sector Neutralisation
- Winsorisation at 1st and 99th percentiles eliminates single-day idiosyncratic outliers.
- Daily cross-sectional OLS regression: Factor = alpha + sum(beta * Sector_dummies) + gamma * Log_Size + epsilon.
- Residuals represent pure idiosyncratic factor exposures free from sector macro tilts.

### Slide 7: Forward Labeling & Loss Formulation
- Target: 21-day forward relative return over the cross-sectional median return.
- Relevance Grades: 5-class quintile partitioning (0 to 4) designed for Normalized Discounted Cumulative Gain (NDCG@10) optimization.

### Slide 8: Purged & Embargoed Cross-Validation
- Standard k-fold leakage breakdown: Multi-day forward holding periods create autoregressive serial correlation.
- Solution: 21-day forward embargo buffer between train and validation time-folds. Zero leakage certified.

### Slide 9: LightGBM LambdaMART Engine
- Pairwise gradient-boosted decision trees optimizing NDCG@10.
- Dynamic group queries mapped per calendar trading date.
- Hyperparameter tuning via Optuna targeting validation Spearman Rank IC.

### Slide 10: Probability Calibration & Reliability Analysis
- Why uncalibrated rank percentiles fail: Ordinal ranks do not reflect true statistical probability.
- Platt Scaling vs. Isotonic Regression benchmarks.
- Expected Calibration Error (ECE) reduction to <0.08.

### Slide 11: Information Coefficient (IC) Analytics Layer
- Daily Spearman Rank IC measurement across the holding horizon.
- Information Ratio of IC (IC-IR): Mean IC / Std IC > 0.40.
- Newey-West adjusted t-statistic > 2.0 confirms statistical significance of manager skill.

### Slide 12: IC Decay Horizon
- Signal decay curve tracked from 1 to 63 trading days.
- Alpha persistence peaks around 15-21 days, validating monthly rebalancing frequency for mutual fund schemes.

### Slide 13: Mondrian Conformal Selection
- Replaces uncalibrated heuristics with distribution-free statistical coverage guarantees (90% confidence, alpha=0.10).
- Sector-conditional non-conformity calibration sets guarantee minimum error bounds per equity sector.

### Slide 14: Decile Backtest & Transaction Cost Modeling
- Decile 10 (Top Conviction) vs Decile 1 (Bottom Conviction) spread analysis.
- 15 bps slippage and transaction drag applied with turnover penalties.
- High Decile Monotonicity confirming monotonic excess returns from D1 to D10.

### Slide 15: Monte Carlo Risk Simulation
- 1,000 simulated 21-day forward paths for shortlisted stocks.
- Estimation of Portfolio Value-at-Risk (VaR 95%) and Conditional VaR (Expected Shortfall).
- Win rate probability estimation across diverse volatility regimes.

### Slide 16: Multi-Model Ensembling & Feature Stability
- Ensemble of LightGBM Ranker, XGBoost Regressor, and Ridge Linear Combiner.
- Cross-sectional Rank-Averaging producing measurable IC Lift.
- Feature stability validation confirming core factors remain stable across market regimes.

### Slide 17: Agentic AI, AWS IaC & Immutable Audit Governance
- Autonomous orchestrator (LangGraph/CrewAI) operating AWS Step Functions, Glue, and SageMaker.
- PII and compliance scanning before shortlist export.
- SHA-256 tamper-evident cryptographic hash-chain preserving immutable audit trail.

### Slide 18: Summary of Institutional Deliverables & Sign-Off
- Deliverables completed: Python Engine, R Replication, Excel Macro Model, Agentic DAG, 35+ Page Report, Model Card.
- Readiness: Ready for immediate integration into Tier 1 Indian Asset Management Company (AMC) equity research desk.
