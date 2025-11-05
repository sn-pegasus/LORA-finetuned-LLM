import argparse
import subprocess
import sys
from pathlib import Path

from .curate import curate_interactions
from .knowledge_store import update_knowledge_from_interactions
from .versioning import record_dataset_snapshot, set_current_adapter_path
from .qa import evaluate_model


def run_training(dataset_path: Path, output_dir: Path, extra_args=None) -> int:
    """Invoke finetune_comprehensive.py with curated dataset to produce a new adapter."""
    cmd = [
        sys.executable,
        str(Path("finetune_comprehensive.py").resolve()),
        "--dataset_path", str(dataset_path),
        "--output_dir", str(output_dir),
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--num_train_epochs", "2",
        "--use_gradient_checkpointing",
        "--bf16",
        "--tf32",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("Launching training:", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser("Continuous Learning Pipeline")
    ap.add_argument("--curate", action="store_true", help="Curate interactions into a ChatML snapshot")
    ap.add_argument("--train", action="store_true", help="Train a new adapter from the latest snapshot")
    ap.add_argument("--eval", action="store_true", help="Run QA evaluation after training")
    ap.add_argument("--output_dir", type=str, default=None, help="Output dir for the new adapter")
    ap.add_argument("--limit", type=int, default=200, help="Max interactions to include in snapshot")
    ap.add_argument("--note", type=str, default="", help="Changelog note")
    ap.add_argument("--eval_path", type=str, default="data/qa_eval_example.jsonl")
    args = ap.parse_args()

    snapshot_path = None
    if args.curate:
        snapshot_path = curate_interactions(limit=args.limit)
        record_dataset_snapshot(str(snapshot_path), note=args.note)
        print("Curated snapshot:", snapshot_path)
        update_knowledge_from_interactions()

    if args.train:
        if snapshot_path is None:
            print("No new snapshot; looking for latest in data/curated...")
            candidates = sorted(Path("data/curated").glob("curated_*.jsonl"))
            if not candidates:
                print("No curated dataset found.")
                sys.exit(1)
            snapshot_path = candidates[-1]
        output_dir = Path(args.output_dir or ("./adapters/adapter_" + snapshot_path.stem))
        output_dir.mkdir(parents=True, exist_ok=True)
        ret = run_training(snapshot_path, output_dir)
        if ret != 0:
            print("Training failed with exit code", ret)
            sys.exit(ret)
        set_current_adapter_path(str(output_dir), note=args.note)
        print("Adapter updated:", output_dir)

    if args.eval:
        res = evaluate_model(eval_path=Path(args.eval_path))
        print("Evaluation:", res)


if __name__ == "__main__":
    main()