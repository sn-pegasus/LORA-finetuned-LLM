import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from .config import REGISTRY_PATH


def _default_registry() -> Dict[str, Any]:
    return {
        "created_ts": time.time(),
        "current_adapter_path": "./llama3b-finetuned",
        "dataset_snapshots": [],
        "adapter_versions": [],
        "changelog": [],
    }


def load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return _default_registry()
    return _default_registry()


def save_registry(reg: Dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def get_current_adapter_path(default: Optional[str] = None) -> str:
    reg = load_registry()
    return reg.get("current_adapter_path", default or "./llama3b-finetuned")


def set_current_adapter_path(path: str, note: str = "") -> None:
    reg = load_registry()
    reg["current_adapter_path"] = path
    reg["adapter_versions"].append({"ts": time.time(), "path": path, "note": note})
    save_registry(reg)


def record_dataset_snapshot(path: str, note: str = "") -> None:
    reg = load_registry()
    reg["dataset_snapshots"].append({"ts": time.time(), "path": path, "note": note})
    save_registry(reg)


def rollback_adapter(steps: int = 1) -> Optional[str]:
    reg = load_registry()
    versions = reg.get("adapter_versions", [])
    if len(versions) <= steps:
        return None
    target = versions[-(steps + 1)]["path"]
    reg["current_adapter_path"] = target
    reg["changelog"].append({
        "ts": time.time(),
        "action": "rollback",
        "to": target,
        "steps": steps,
    })
    save_registry(reg)
    return target