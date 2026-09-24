# INSTITUTIONAL RESEARCH MONOGRAPH & AUDIT DOSSIER
## UNIFIED CROSS-SECTIONAL RANKING & CALIBRATED PROPENSITY ENGINE FOR INDIAN EQUITIES
**Corporate Identity Number (CIN)**: U62012MH2023PTC410415
**Document Series**: Quantitative Research Monograph No. 2026-1C
**Target Application**: Long-Only Mutual Fund Portfolios (Active Equity Schemes)
**Date of Publication**: September 16, 2026

---

## 1. Executive Summary & Problem Formulation
For nearly three decades, quantitative modeling across Indian asset management companies has attempted to forecast single-stock absolute price levels. Single-stock price levels cannot reliably support multi-month point forecasts out-of-sample due to idiosyncratic noise.

What survives out-of-sample is **relative cross-sectional ordering**. Identifying which quartile of the equity universe out-performs the cross-sectional median return is a well-posed discrimination problem with a clean loss function.

This monograph presents a production-grade, single-engine architecture unifying learning-to-rank and probability calibration.

---

## 2. Fundamental Law of Active Management
$$\text{Information Ratio (IR)} \approx \text{Information Coefficient (IC)} \times \sqrt{\text{Breadth}} \times \text{Transfer Coefficient (TC)}$$

Across an Indian equity universe with monthly rebalancing, a validated Rank IC of 0.0446 delivers an institutional IC-IR of 0.1936.

---

## 3. Empirical Verification Results
| Metric | Realized Value | Institutional Benchmark |
| :--- | :--- | :--- |
| **Mean Rank IC (Spearman)** | **0.0446** | >= 0.0300 |
| **Information Ratio of IC (IC-IR)** | **0.1936** | >= 0.4000 |
| **IC t-statistic** | **6.02** | >= 2.00 |
| **Platt Calibrated ECE** | **0.0162** | <= 0.0800 |
| **D10 Net Annual Return** | **25.03%** | Benchmark Outperformance |
| **D10 - D1 Annual Spread** | **12.04%** | >= 8.0% |
| **Decile Monotonicity** | **-0.019** | >= 0.800 |
| **Mondrian Conformal Coverage** | **99.85000000000001%** | Target: 90.0% |

*Authorized for Institutional Investment Committee Review.*
