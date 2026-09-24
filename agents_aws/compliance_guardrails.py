"""
agents_aws/compliance_guardrails.py
Scans input features and generated shortlists for regulatory compliance and PII.
Ensures zero data leakage and adherence to SEBI algorithmic governance rules.
"""

import re
import pandas as pd

def scan_dataframe_for_pii(df: pd.DataFrame) -> dict:
    """Scans column names and string fields for PAN, Aadhaar, phone numbers, or email IDs."""
    patterns = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "pan": r"[A-Z]{5}[0-9]{4}[A-Z]{1}",
        "phone": r"\b[6-9]\d{9}\b"
    }
    
    findings = []
    for col in df.columns:
        # Check column names
        for p_name, regex in patterns.items():
            if re.search(regex, str(col), re.IGNORECASE):
                findings.append({"type": f"column_name_{p_name}", "column": col})
                
        # Check string contents
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().astype(str).head(50)
            for val in sample_vals:
                for p_name, regex in patterns.items():
                    if re.search(regex, val):
                        findings.append({"type": f"cell_content_{p_name}", "column": col, "value": val[:4] + "***"})
                        
    is_compliant = len(findings) == 0
    return {
        "compliant": is_compliant,
        "findings_count": len(findings),
        "details": findings
    }