import json
from pathlib import Path
from typing import Dict

from .config import LOGS_DIR

AGG_PATH = LOGS_DIR / "feedback_metrics.json"


def _load() -> Dict:
    if AGG_PATH.exists():
        try:
            return json.loads(AGG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(obj: Dict) -> None:
    AGG_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGG_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def record_feedback(value: str) -> None:
    agg = _load()
    if "counts" not in agg:
        agg["counts"] = {"positive": 0, "negative": 0, "neutral": 0}
    if value not in agg["counts"]:
        value = "neutral"
    agg["counts"][value] += 1
    _save(agg)