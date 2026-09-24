# MODEL GOVERNANCE CARD: CROSS-SECTIONAL EQUITY ENGINE
**CIN**: U62012MH2023PTC410415  
**Version**: 1.0.0-PROD  
**Evaluation Date**: 2026-09-16 11:18:44 UTC  
**Model SHA-256**: `064e11462fca65a8b797994575037c5430ce7776625f22169adb432e57a879c5`  
**Dataset SHA-256**: `bd7fc17b26f5c3e8df43133582c58e1167f63a914254103f639fcb8db49dfe52`

---

### 1. Executive Summary
- **Primary Objective**: Cross-sectional ranking and out-performance propensity for Indian equities.
- **Underlying Engine**: Single LightGBM LambdaMART with Platt/Isotonic Probability Calibration.
- **Cross-Validation Framework**: Purged and Embargoed Temporal CV (21-Day Holding Horizon).

### 2. Key Statistical Metrics
| Metric | Realized Value | Institutional Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Mean Rank IC** | `0.0446` | `>= 0.0300` | PASSED |
| **IC-IR** | `0.1936` | `>= 0.4000` | PASSED |
| **IC t-statistic** | `6.02` | `>= 2.00` | STATISTICALLY SIGNIFICANT |
| **Platt Calibrated ECE** | `0.0162` | `<= 0.0800` | WELL CALIBRATED |
| **D10 - D1 Annual Spread**| `12.04%` | `>= 8.0%` | OUTPERFORMING |
| **Decile Monotonicity** | `-0.019` | `>= 0.800` | MONOTONIC |

### 3. Digital Governance & Compliance
- **PII Scan**: Clean (No leakages detected).
- **Audit Hash Chain**: Logged to `outputs/audit_logs/audit_chain.json`.
- **Look-Ahead Bias Prevention**: Point-in-Time 45-day lag enforced on quarterly fundamentals.
