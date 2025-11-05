import os
from pathlib import Path

# Base directories
# Resolve to the project root regardless of current working directory
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INGEST_DIR = DATA_DIR / "ingestion"
CURATED_DIR = DATA_DIR / "curated"
LOGS_DIR = BASE_DIR / "logs"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
REGISTRY_PATH = BASE_DIR / "continuous_learning" / "registry.json"

# Privacy salt for hashing user identifiers
SALT_ENV = os.getenv("CL_USER_HASH_SALT", "CHANGE_ME_SALT")

# Ensure directories exist
for p in [INGEST_DIR, CURATED_DIR, LOGS_DIR, KNOWLEDGE_DIR]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Best-effort; avoid crashing chat
        pass