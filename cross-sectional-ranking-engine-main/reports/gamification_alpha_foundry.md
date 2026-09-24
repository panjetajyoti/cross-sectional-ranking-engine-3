# ALPHA FOUNDRY: GAMIFIED QUANTITATIVE SIMULATION SPECIFICATION
**CIN**: U62012MH2023PTC410415
**Domain**: Cross-Sectional Stock Selection & Factor Engineering
**Role**: Junior Buy-Side Quantitative Analyst

---

## 1. Executive Concept
Alpha Foundry simulates a 15-day institutional desk rotation inside a Tier 1 Indian Asset Management Company (AMC). The player progresses through 8 progressive operational clearance levels, facing realistic Indian market anomalies, SEBI regulatory compliance checks, and cross-sectional liquidity hurdles.

---

## 2. The 8 Progression Levels
1. **Level 1: The Bhavcopy Maze**
   *Objective*: Ingest 5+ years of raw NSE corporate actions and bhavcopies without forward leakage.
   *Failure Condition*: Look-ahead contamination in earnings announcement dates.

2. **Level 2: The Factor Forge**
   *Objective*: Construct 6 core factor families (Momentum, Value, Quality, Low-Risk, Size, Flow) and cross-sectionally neutralise them against Sector dummies and Market Cap.

3. **Level 3: The Cross-Sectional Crucible**
   *Objective*: Replace point-prediction targets with 21-day forward relative return splits and relevance grades.

4. **Level 4: The Lambda Colosseum**
   *Objective*: Train a LightGBM LambdaMART engine using Purged and Embargoed temporal splits.

5. **Level 5: The Calibration Chamber**
   *Objective*: Pass through Platt and Isotonic calibration to bring Expected Calibration Error (ECE) below 0.08.

6. **Level 6: The Conformal Shield**
   *Objective*: Wrap shortlist predictions with Mondrian conformal bounds guaranteeing 90% sector coverage.

7. **Level 7: The Execution Arena**
   *Objective*: Run a 21-day rebalanced decile backtest net of 15 bps slippage and transaction costs. Achieve Decile Monotonicity > 0.80.

8. **Level 8: The Investment Committee Audit**
   *Objective*: Defend the model before the Head of Research with SHA-256 cryptographic logs, R replication checks, and an automated Model Governance Card.

---

## 3. The 5 Stress Scenarios
1. **The Adani Hindenburg Shock (January 2023)**: Extreme idiosyncratic volatility spike; tests downside vol filter.
2. **The Post-COVID Small-Cap Frenzy (2021-2022)**: Momentum crash and mean reversion; tests 12M-1M skip-month protection.
3. **The Budget Day Volatility Spike**: Intraday liquidity freeze; tests turnover penalties and 15 bps cost modeling.
4. **The RBI Surprise Rate Hike**: Sector-wide banking re-rating; verifies sector residualisation neutralises interest rate beta.
5. **The SEBI Regulatory Scrutiny**: Audit log verification drill; tests cryptographic hash-chain integrity.
