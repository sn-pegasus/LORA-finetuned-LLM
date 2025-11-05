import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import INGEST_DIR, CURATED_DIR


DEFAULT_SYSTEM_PROMPT = (
    "You are a pharmaceutical batch record assistant. Provide precise, factual "
    "answers based only on the batch data. Keep responses concise and accurate."
)


def _load_raw(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = INGEST_DIR / "raw_interactions.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
            except Exception:
                continue
    return rows


def _chatml_example(question: str, answer: str) -> Dict[str, str]:
    text = (
        f"<|system|>\n{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"<|user|>\n{question}\n\n"
        f"<|assistant|>\n{answer}\n<|end|>"
    )
    return {"text": text}


def curate_interactions(min_feedback: str = "neutral", limit: Optional[int] = None) -> Path:
    """Build a curated ChatML dataset snapshot from raw interactions.

    Filters by feedback and removes entries with PII flags.
    """
    raw = _load_raw(limit=limit)
    if not raw:
        raise RuntimeError("No raw interactions available to curate.")

    wanted = {"positive", "neutral"} if min_feedback == "neutral" else {"positive"}
    seen_ids = set()
    out_rows: List[Dict[str, str]] = []
    for r in raw:
        if r.get("feedback", "neutral") not in wanted:
            continue
        # Exclude entries with PII redaction flags
        if r.get("flags"):
            continue
        if r.get("dedup_id") in seen_ids:
            continue
        seen_ids.add(r.get("dedup_id"))
        out_rows.append(_chatml_example(r.get("question", ""), r.get("answer", "")))

    if not out_rows:
        raise RuntimeError("No interactions passed curation filters.")

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = CURATED_DIR / f"curated_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_path