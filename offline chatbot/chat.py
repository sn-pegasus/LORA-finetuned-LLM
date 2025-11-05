#!/usr/bin/env python3
"""
Simple chat interface for the trained LoRA model
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    print("="*60)
    print("OFFLINE CHATBOT - Interactive Chat")
    print("="*60)
    print("\n[*] Loading model (this takes about 60-90 seconds)...")
    
    adapter_path = "./llama3b-finetuned"
    base_model_id = "meta-llama/Llama-3.2-3B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens (same as during training)
    special_tokens = ["<|user|>", "<|assistant|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    
    # Resize embeddings and load LoRA adapter if available; otherwise use base model
    import os
    base_model.resize_token_embeddings(len(tokenizer))
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    config_file = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_file) and os.path.exists(config_file):
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("[!] Adapter not found. Running with base model only (CPU).")
        model = base_model
    model.eval()
    
    print(f"[OK] Model loaded successfully!")
    print("\n" + "="*60)
    print("PHARMACEUTICAL LOT ASSISTANT")
    print("Ask questions about lot numbers, batch records, etc.")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'stop', 'q']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
            
            # Format as chat
            formatted_input = f"<|user|>\n{user_input}\n\n<|assistant|>\n"
            
            # Tokenize
            inputs = tokenizer(formatted_input, return_tensors="pt")
            
            # Generate
            print("Assistant: ", end="", flush=True)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract only the assistant's response (after the last <|assistant|> tag)
            if "<|assistant|>" in full_output:
                parts = full_output.split("<|assistant|>")
                assistant_response = parts[-1].strip()
            else:
                # Fallback: remove the input prompt
                assistant_response = full_output[len(formatted_input):].strip()
            
            # Remove the user's question if it's being repeated
            if assistant_response.lower().startswith(user_input.lower()):
                assistant_response = assistant_response[len(user_input):].strip()
            
            # Print the response (stop at first occurrence of <|user|> if model continues)
            if "<|user|>" in assistant_response:
                assistant_response = assistant_response.split("<|user|>")[0].strip()
            
            print(assistant_response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            continue

if __name__ == "__main__":
    main()
