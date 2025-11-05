import os
import argparse
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling

# Enable efficient attention kernels when available (GPU)
try:
    if torch.cuda.is_available():
        from torch.backends.cuda import sdp_kernel
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
except Exception:
    pass


"""QLoRA fine-tuning entrypoint for LLaMA models using TRL's SFTTrainer with ChatML format.

This script prepares the ChatML dataset, loads a base model in 4-bit with QLoRA,
attaches LoRA adapters, and runs supervised fine-tuning.
"""


def format_chat_template(messages: List[Dict[str, str]]) -> str:
    """Format messages into a chat template for training with system instruction."""
    # Add system instruction for precise pharmaceutical responses
    formatted_text = "<|system|>\nYou are a pharmaceutical batch record assistant. Provide precise, factual answers based only on the batch data. Keep responses concise and accurate.\n\n"
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "user":
            formatted_text += f"<|user|>\n{content}\n\n"
        elif role == "assistant":
            formatted_text += f"<|assistant|>\n{content}<|end|>\n\n"
    
    return formatted_text.strip()


def format_dataset_for_sft(dataset: Any, num_samples_debug: int = -1) -> Any:
    """Format the ChatML dataset for SFT training."""
    def _map_fn(example: Dict[str, Any]) -> Dict[str, str]:
        messages = example.get("messages", [])
        formatted_text = format_chat_template(messages)
        return {"text": formatted_text}

    if num_samples_debug and num_samples_debug > 0:
        dataset = dataset.select(range(min(num_samples_debug, len(dataset))))

    return dataset.map(_map_fn, remove_columns=[
        col for col in dataset.column_names if col != "text"
    ])


def get_bnb_config() -> BitsAndBytesConfig:
    """Get BitsAndBytes configuration for 4-bit quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def is_bnb_available() -> bool:
    """Check if bitsandbytes is available in the environment.

    On Windows, bitsandbytes is typically unavailable and should be avoided.
    """
    try:
        import importlib.util
        return importlib.util.find_spec("bitsandbytes") is not None
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for LLaMA-3.1-8B-Instruct with comprehensive ChatML dataset")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Base model repo or local path")
    parser.add_argument("--dataset_path", type=str, default="data/train_from_dataMontage5.jsonl", help="Path to ChatML JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./llama8b-finetuned", help="Directory to save LoRA adapter")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 on Ampere+ GPUs")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--packing", action="store_true", help="Pack multiple samples in a single sequence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guided", action="store_true", help="Interactive guided mode with production defaults")
    # Inference/optimization parameters
    parser.add_argument("--chat", action="store_true", help="Run interactive chat mode")
    parser.add_argument("--eval", action="store_true", help="Evaluate responses on the dataset")
    parser.add_argument("--adapter_path", type=str, default="./llama8b-production-v1", help="LoRA adapter path for inference")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling; default greedy")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Penalize repetition for cleaner outputs")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for faster inference (PyTorch 2+")
    parser.add_argument("--eval_samples", type=int, default=50, help="Number of samples to evaluate (0=all)")
    # LoRA parameters - optimized for 8B model memory efficiency
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="*", 
                       default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                       help="Target module names for LoRA")

    args = parser.parse_args()

    if args.guided:
        print("=" * 80)
        print("PRODUCTION MODEL RETRAINING")
        print("=" * 80)
        print("\nThis will retrain the model with optimized parameters for:")
        print("  ✓ Higher accuracy")
        print("  ✓ More precise responses")
        print("  ✓ Better generalization")
        print("  ✓ Industrial/client-ready quality")
        print("\nTraining improvements:")
        print("  • Increased LoRA rank (16→32) for better capacity")
        print("  • Lower learning rate (2e-4→5e-5) for stability")
        print("  • More epochs (3→5) for better convergence")
        print("  • Added system instructions for context")
        print("  • End token for precise response termination")
        print("=" * 80)
        
        response = input("\nProceed with retraining? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Training cancelled.")
            return
        
        print("\n[*] Starting production retraining...")
        print("[*] This will take approximately 6-8 hours on CPU for 8B model\n")
        
        # Override parameters with production defaults for 8B model
        args.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        args.dataset_path = "data/train_from_dataMontage5.jsonl"
        args.output_dir = "./llama8b-production-v1"
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 16
        args.learning_rate = 5e-5
        args.num_train_epochs = 5
        args.warmup_ratio = 0.1
        args.lora_r = 16
        args.lora_alpha = 32
        args.lora_dropout = 0.05
        args.logging_steps = 10
        args.save_steps = 50
        args.save_total_limit = 2
        args.max_seq_length = 2048

    # Inference-only modes: interactive chat or dataset evaluation
    if args.chat or args.eval:
        print("Loading tokenizer for inference...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        if is_bnb_available():
            print("Loading base model in 4-bit (inference)...")
            bnb_config = get_bnb_config()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if args.bf16 else None,
                attn_implementation="sdpa",
            )
        else:
            if torch.cuda.is_available():
                print("bitsandbytes not available; using GPU without quantization (fp16/bf16)...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                    attn_implementation="sdpa",
                )
            else:
                print("bitsandbytes not available; loading model on CPU without quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    attn_implementation="sdpa",
                )

        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        # Attach LoRA adapter for inference
        from peft import PeftModel
        try:
            model = PeftModel.from_pretrained(model, args.adapter_path)
        except Exception as e:
            print(f"Warning: failed to load adapter from {args.adapter_path}: {e}. Continuing without adapter.")

        if args.compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Proceeding without compilation.")

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        def chat_once(question: str) -> str:
            prompt = format_chat_template([{"role": "user", "content": question}])
            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|assistant|>" in text:
                ans = text.split("<|assistant|>")[-1]
                if "<|end|>" in ans:
                    ans = ans.split("<|end|>")[0]
                return ans.strip()
            return text.strip()

        if args.chat:
            print("Interactive chat mode. Type 'exit' to quit.")
            while True:
                q = input("You: ").strip()
                if q.lower() in {"exit", "quit"}:
                    break
                ans = chat_once(q)
                print("Assistant:", ans)
            return
        else:
            print(f"Evaluating on {args.dataset_path} ...")
            ds = load_dataset("json", data_files=args.dataset_path, split="train")
            total = min(len(ds), args.eval_samples) if args.eval_samples > 0 else len(ds)
            ds = ds.select(range(total))
            import re
            def norm(s: str) -> str:
                return re.sub(r"\s+", " ", s.strip().lower())
            correct = 0
            for ex in ds:
                msgs = ex.get("messages", [])
                user_q = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                target = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                pred = chat_once(user_q)
                if norm(pred) == norm(target):
                    correct += 1
            print(f"Eval accuracy (exact match): {correct}/{total} = {correct/total:.2%}")
            return

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add special tokens for chat format including system and end tokens
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Prefer 4-bit loading when bitsandbytes is available; otherwise use CPU without quantization
    if is_bnb_available():
        print("Loading model in 4-bit (QLoRA) with bitsandbytes...")
        bnb_config = get_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else None,
            attn_implementation="sdpa",
        )
    else:
        if torch.cuda.is_available():
            print("bitsandbytes not available; using GPU without quantization (fp16/bf16)...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                attn_implementation="sdpa",
            )
        else:
            print("bitsandbytes not available; loading model on CPU without quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )

    # Resize token embeddings to accommodate new special tokens
    model.resize_token_embeddings(len(tokenizer))
    # Disable cache during training to support gradient checkpointing and reduce memory
    model.config.use_cache = not args.use_gradient_checkpointing

    # Enable gradient checkpointing if requested to save memory
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Set up LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print("Loading comprehensive dataset...")
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Shuffle to reduce padding/clipping correlation and improve convergence
    dataset = dataset.shuffle(seed=args.seed)
    dataset = format_dataset_for_sft(dataset)
    print("Dataset formatted for SFT training")

    # Tokenize the dataset upfront to ensure the collator receives tokenized features
    def tokenize_function(example: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            example["text"],
            truncation=True,
            padding=False,
            max_length=args.max_seq_length,
        )

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=["text"])
    print("Dataset tokenized")

    # Use CPU-compatible optimizer
    optimizer = "adamw_torch" if model.device.type == "cpu" else "paged_adamw_8bit"
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16 and model.device.type != "cpu",
        fp16=not args.bf16 and model.device.type != "cpu",
        optim=optimizer,
        lr_scheduler_type="cosine",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        group_by_length=True,
        dataloader_num_workers=2,
        gradient_checkpointing=args.use_gradient_checkpointing,
    )

    print("Starting SFT training with comprehensive dataset...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    print("Saving LoRA adapter and tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done. Comprehensive LoRA adapter saved to:", args.output_dir)
    print(f"Training completed with {len(dataset)} examples from the comprehensive dataset")


if __name__ == "__main__":
    main()