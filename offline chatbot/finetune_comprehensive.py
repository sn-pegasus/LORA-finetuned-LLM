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


"""QLoRA fine-tuning entrypoint for LLaMA models using TRL's SFTTrainer with ChatML format.

This script prepares the ChatML dataset, loads a base model in 4-bit with QLoRA,
attaches LoRA adapters, and runs supervised fine-tuning.
"""


def format_chat_template(messages: List[Dict[str, str]]) -> str:
    """Format messages into a chat template for training."""
    formatted_text = ""
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "user":
            formatted_text += f"<|user|>\n{content}\n\n"
        elif role == "assistant":
            formatted_text += f"<|assistant|>\n{content}\n\n"
    
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
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for LLaMA-3B with comprehensive ChatML dataset")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B", help="Base model repo or local path")
    parser.add_argument("--dataset_path", type=str, default="data/train_comprehensive.jsonl", help="Path to ChatML JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./llama3b-finetuned", help="Directory to save LoRA adapter")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 on Ampere+ GPUs")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--packing", action="store_true", help="Pack multiple samples in a single sequence")
    parser.add_argument("--seed", type=int, default=42)
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, nargs="*", 
                       default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                       help="Target module names for LoRA")

    args = parser.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add special tokens for chat format
    special_tokens = ["<|user|>", "<|assistant|>"]
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
        )
    else:
        print("bitsandbytes not available; loading model on CPU without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    # Resize token embeddings to accommodate new special tokens
    model.resize_token_embeddings(len(tokenizer))

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