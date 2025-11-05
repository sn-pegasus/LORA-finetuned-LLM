Continuous Learning System
==========================

Overview
--------
- Securely ingests user interactions, sanitizes PII, and stores raw logs.
- Curates clean interactions into ChatML snapshots for incremental fine-tuning.
- Maintains a version registry of dataset snapshots and adapter paths, enabling rollback.
- Tracks performance and feedback to ensure quality does not degrade.
- Provides a pipeline CLI to curate → train → evaluate → update adapter.

Directories
-----------
- `data/ingestion/raw_interactions.jsonl`: sanitized raw interaction logs
- `data/curated/curated_YYYYMMDD_HHMMSS.jsonl`: curated ChatML snapshots
- `knowledge/knowledge.json`: simple aggregated facts (e.g., lot mentions)
- `logs/learning_events.jsonl`: ingestion/cycle events
- `logs/generation_metrics.jsonl`: latency and tokens/sec for chat generations
- `logs/feedback_metrics.json`: user feedback counters
- `continuous_learning/registry.json`: adapter and snapshot registry

Usage
-----
1) Interact and provide feedback in `chat.py`:
   - After each answer, enter `y`/`n` to mark helpfulness (or press Enter to skip).
   - Interactions are sanitized and logged automatically.

2) Curate and train a new adapter:
   - `python -m continuous_learning.pipeline --curate --train --eval --note "weekly update"`
   - This creates a snapshot from raw logs, trains an adapter using `finetune_comprehensive.py`,
     runs a lightweight QA eval, and updates the current adapter path.

3) Roll back adapter (manual):
   - Use `continuous_learning.versioning.rollback_adapter(steps=1)` in a Python REPL,
     or directly edit `continuous_learning/registry.json` to a previous `adapter_versions` entry.
   - Restart `chat.py` to use the rolled-back adapter.

Security & Privacy
------------------
- Inputs are validated and sanitized for common PII (emails, phones, SSNs, cards).
- User identifiers are hashed using `CL_USER_HASH_SALT` (set in your environment).
- All artifacts remain local; no external services are called.

Quality Assurance
-----------------
- Feedback-driven curation excludes entries with PII flags and prioritizes helpful interactions.
- QA checks measure keyword-based accuracy on an eval set and track latency/tokens/sec.
- Performance monitoring logs metrics for regression detection.

Testing Procedures
------------------
- Ingestion: manually verify `raw_interactions.jsonl` contains sanitized entries without PII.
- Curation: run pipeline and confirm `curated_*.jsonl` is generated with ChatML examples.
- Evaluation: prepare `data/qa_eval_example.jsonl` with `{prompt, keywords}` pairs; run `--eval`.
- Rollback: update registry via `rollback_adapter`, restart `chat.py`, and confirm adapter changes.
- Scaling: increase usage (interactions), confirm curation and training proceed without issues.

Notes
-----
- On Windows without bitsandbytes, training runs with GPU `fp16/bf16` fallback when possible.
- For larger updates, consider WSL2 + CUDA to enable 4-bit QLoRA for faster fine-tuning.