import json
import time
from pathlib import Path
from typing import Dict, Any

from .config import LOGS_DIR


GEN_PATH = LOGS_DIR / "generation_metrics.jsonl"


def log_generation(stats: Dict[str, Any]) -> None:
    entry = {"ts": time.time(), **stats}
    GEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GEN_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")