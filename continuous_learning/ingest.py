import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from .config import INGEST_DIR, LOGS_DIR, SALT_ENV
from .privacy import sanitize_text, validate_input


RAW_PATH = INGEST_DIR / "raw_interactions.jsonl"
EVENTS_PATH = LOGS_DIR / "learning_events.jsonl"


def _hash_user_id(user_id: str) -> str:
    salted = (SALT_ENV + ":" + (user_id or "anonymous")).encode("utf-8")
    return hashlib.sha256(salted).hexdigest()


def _dedup_id(question_clean: str, answer_clean: str) -> str:
    payload = (question_clean + "\n" + answer_clean).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    entry = {
        "ts": time.time(),
        "event_type": event_type,
        "payload": payload,
    }
    try:
        _append_jsonl(EVENTS_PATH, entry)
    except Exception:
        # Avoid crashing chat pipeline on logging failure
        pass


def record_interaction(
    user_id: str,
    question: str,
    answer: str,
    feedback: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a single chat interaction with privacy sanitization and logging.

    Returns the stored record.
    """
    if not (validate_input(question) and validate_input(answer)):
        raise ValueError("Invalid question/answer inputs")

    q = sanitize_text(question)
    a = sanitize_text(answer)
    entry = {
        "ts": time.time(),
        "user_id_hash": _hash_user_id(user_id or "anonymous"),
        "question": q["text"],
        "answer": a["text"],
        "flags": sorted(list(set(q["flags"] + a["flags"]))),
        "feedback": feedback or "neutral",
        "metadata": metadata or {},
    }
    entry["dedup_id"] = _dedup_id(entry["question"], entry["answer"])

    _append_jsonl(RAW_PATH, entry)
    log_event("ingest", {"dedup_id": entry["dedup_id"], "flags": entry["flags"], "feedback": entry["feedback"]})
    return entry