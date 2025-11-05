import json
import re
from pathlib import Path
from typing import Dict, Any, List

from .config import KNOWLEDGE_DIR, INGEST_DIR


LOT_RE = re.compile(r"\b[Ll](\d{3,5})\b")


def _load_raw() -> List[Dict[str, Any]]:
    path = INGEST_DIR / "raw_interactions.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def update_knowledge_from_interactions() -> Path:
    """Extract simple facts (lot numbers mentions) to a knowledge store."""
    rows = _load_raw()
    store_path = KNOWLEDGE_DIR / "knowledge.json"
    knowledge: Dict[str, Any] = {"lots": {}, "stats": {"mentions": 0}}
    if store_path.exists():
        try:
            knowledge = json.loads(store_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    for r in rows:
        text = (r.get("question", "") + "\n" + r.get("answer", "")).lower()
        for m in LOT_RE.finditer(text):
            lot_id = f"L{m.group(1)}".upper()
            knowledge["lots"].setdefault(lot_id, {"mentions": 0})
            knowledge["lots"][lot_id]["mentions"] += 1
            knowledge["stats"]["mentions"] += 1

    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
    return store_path