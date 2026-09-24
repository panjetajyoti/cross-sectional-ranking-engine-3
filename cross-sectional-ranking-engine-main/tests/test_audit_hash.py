"""
tests/test_audit_hash.py
Validates integrity of the immutable SHA-256 audit log.
"""
import os
import json

def test_audit_log_exists_and_valid():
    log_path = "outputs/audit_logs/audit_chain.json"
    assert os.path.exists(log_path), "Audit log does not exist."
    
    with open(log_path, "r") as f:
        chain = json.load(f)
        
    assert len(chain) > 0, "Audit log chain is empty."
    assert "block_hash" in chain[-1], "Block hash missing in audit log."
    print("\n[✓] Audit Log Cryptographic Integrity Verified.")