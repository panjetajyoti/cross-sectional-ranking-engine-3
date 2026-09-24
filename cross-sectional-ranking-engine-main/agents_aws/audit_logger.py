"""
agents_aws/audit_logger.py
Cryptographic SHA-256 tamper-evident hash-chain audit logging.
Maintains an immutable chain of custody for every model run and dataset version.
"""

import os
import json
import hashlib
from datetime import datetime

AUDIT_LOG_PATH = "outputs/audit_logs/audit_chain.json"

def calculate_sha256(file_path: str) -> str:
    """Calculates cryptographic hash of any artifact."""
    if not os.path.exists(file_path):
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def append_audit_event(action: str, artifact_path: str, metadata: dict = None) -> dict:
    """Appends a new verified event block to the SHA-256 hash-chain."""
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    
    chain = []
    prev_hash = "GENESIS_BLOCK_00000000000000000000000000000000000000000000000000000000"
    
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            with open(AUDIT_LOG_PATH, "r") as f:
                chain = json.load(f)
            if len(chain) > 0:
                prev_hash = chain[-1]["block_hash"]
        except Exception:
            chain = []

    artifact_hash = calculate_sha256(artifact_path)
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    block_payload = {
        "index": len(chain) + 1,
        "timestamp": timestamp,
        "action": action,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_hash,
        "metadata": metadata or {},
        "previous_hash": prev_hash
    }
    
    # Calculate current block hash (payload + prev_hash)
    payload_str = json.dumps(block_payload, sort_keys=True)
    current_hash = hashlib.sha256(payload_str.encode()).hexdigest()
    block_payload["block_hash"] = current_hash
    
    chain.append(block_payload)
    
    with open(AUDIT_LOG_PATH, "w") as f:
        json.dump(chain, f, indent=4)
        
    print(f"[Audit Chain] Block #{block_payload['index']} recorded: {action} -> Hash: {current_hash[:12]}...")
    return block_payload

if __name__ == "__main__":
    append_audit_event("INIT_AUDIT", "outputs/audit_logs/audit_chain.json", {"status": "initialized"})