import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from .versioning import get_current_adapter_path


def _load_eval(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
            except Exception:
                continue
    return rows


def _contains_all(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return all(k.lower() in t for k in keywords)


def evaluate_model(
    base_model_id: str = "meta-llama/Llama-3.2-3B",
    eval_path: Path = Path("data/qa_eval_example.jsonl"),
    limit: Optional[int] = 25,
) -> Dict[str, Any]:
    """Run a lightweight keyword-based evaluation on an example set.

    Each eval item should be {"prompt": str, "keywords": [str]}
    """
    data = _load_eval(eval_path, limit=limit)
    if not data:
        return {"ok": False, "reason": f"No eval data at {eval_path}"}

    adapter_path = get_current_adapter_path()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel.enable_flash_sdp(True)
            sdp_kernel.enable_mem_efficient_sdp(True)
            sdp_kernel.enable_math_sdp(False)
        except Exception:
            pass
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16,
            attn_implementation="sdpa",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    results = {"correct": 0, "total": 0, "latency_ms": [], "tokens_per_s": []}
    for item in data:
        prompt = item.get("prompt", "")
        keywords = item.get("keywords", [])
        formatted = (
            "<|system|>\nYou are a pharmaceutical batch record assistant.\n\n"
            f"<|user|>\n{prompt}\n\n<|assistant|>\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            end_id = tokenizer.convert_tokens_to_ids("<|end|>")
            eos_ids = [tokenizer.eos_token_id] + ([end_id] if end_id is not None else [])
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
        dt = time.time() - t0
        out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_len = outputs[0].shape[0] - inputs["input_ids"].shape[1]
        if dt > 0:
            results["tokens_per_s"].append(gen_len / dt)
        results["latency_ms"].append(dt * 1000.0)
        results["total"] += 1
        if _contains_all(out_text, keywords):
            results["correct"] += 1

    acc = (results["correct"] / results["total"]) if results["total"] else 0.0
    return {
        "ok": True,
        "accuracy": acc,
        "avg_latency_ms": sum(results["latency_ms"]) / len(results["latency_ms"]) if results["latency_ms"] else None,
        "avg_tokens_per_s": sum(results["tokens_per_s"]) / len(results["tokens_per_s"]) if results["tokens_per_s"] else None,
        "details": results,
    }