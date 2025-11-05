#!/usr/bin/env python3
"""
Simple chat interface for the trained LoRA model
"""
import sys
import time
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
try:
    from continuous_learning.versioning import get_current_adapter_path
    from continuous_learning.ingest import record_interaction
    from continuous_learning.feedback import record_feedback
    from continuous_learning.monitoring import log_generation
except Exception:
    # Optional fallback if continuous_learning modules are unavailable
    get_current_adapter_path = lambda default=None: default or "./llama8b-production-v1"
    def record_interaction(*args, **kwargs):
        return {}
    def record_feedback(*args, **kwargs):
        return None
    def log_generation(*args, **kwargs):
        return None

def main():
    print("="*60)
    print("OFFLINE CHATBOT - Interactive Chat")
    print("="*60)
    print("\n[*] Loading model (this takes about 60-90 seconds)...")
    
    adapter_path = get_current_adapter_path("./llama8b-production-v1/checkpoint-55")
    # Auto-detect base model from adapter_config.json, fallback to 3B
    base_model_id = "meta-llama/Llama-3.2-3B"
    try:
        import json, os
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                bm = cfg.get("base_model_name_or_path")
                if isinstance(bm, str) and bm.strip():
                    base_model_id = bm.strip()
    except Exception:
        pass
    
    # Load tokenizer from adapter path (includes special tokens)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
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
    
    # Resize embeddings to match the fine-tuned tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter
    print("[*] Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print(f"[OK] Model loaded successfully!")
    print("\n" + "="*60)
    print("PHARMACEUTICAL LOT ASSISTANT")
    print("Ask questions about lot numbers, batch records, etc.")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    # Privacy-preserving in-session memory (not persisted)
    session_state = {"user_name": None}

    def update_name_from_text(text: str):
        try:
            t = text or ""
            # Clear name commands
            if re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
                session_state["user_name"] = None
                return None
            # Set name commands (avoid matching if a forget intent is present)
            if not re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
                m = re.search(r"\b(?:remember|remeber|rememeber)?\s*my\s+name\s+(?:is|as)\s+([A-Za-z][A-Za-z .'-]*)", t, re.IGNORECASE)
                if m:
                    name = m.group(1).strip()
                    session_state["user_name"] = name
                    return name
        except Exception:
            pass
        return None

    def memory_command_reply(text: str):
        try:
            t = text or ""
            if re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
                session_state["user_name"] = None
                return "Okay, I forgot your name for this session."
            m = re.search(r"\b(?:remember|remeber|rememeber)?\s*my\s+name\s+(?:is|as)\s+([A-Za-z][A-Za-z .'-]*)", t, re.IGNORECASE)
            if m and not re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
                name = m.group(1).strip()
                session_state["user_name"] = name
                return f"Got it. I'll remember your name is {name} for this session."
        except Exception:
            pass
        return None

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'stop', 'q']:
                print("\nGoodbye!")
                break
                
            if not user_input:
                continue
            
            # Intercept memory commands (set/forget) before generation
            mem_reply = memory_command_reply(user_input)
            if mem_reply is not None:
                print(mem_reply)
                print()
                fb = input("Feedback (y=helpful / n=not helpful / Enter=skip): ").strip().lower()
                feedback = "positive" if fb == "y" else ("negative" if fb == "n" else "neutral")
                # Update memory via feedback if present
                update_name_from_text(fb)
                try:
                    record_feedback(feedback)
                except Exception:
                    pass
                try:
                    record_interaction(user_id="local_user", question=user_input, answer=mem_reply, feedback=feedback)
                    log_generation({
                        "latency_ms": 0.0,
                        "tokens_per_s": 0.0,
                        "gen_len": 0,
                    })
                except Exception:
                    pass
                continue

            # Update session memory silently for "my name is ..." (without immediate reply)
            _ = update_name_from_text(user_input)

            # Format as chat with system instruction + memory note + guardrails
            system_inst = (
                "You are a pharmaceutical batch record assistant. Provide precise, factual answers based only on the batch data. "
                "Keep responses concise and accurate. Never invent personal identity or names. "
                "If asked about the user's name, only answer using the session memory. "
                "If the name is not set, state that you don't know."
            )
            memory_note = f" Known user name: {session_state['user_name']}" if session_state['user_name'] else ""
            formatted_input = f"<|system|>\n{system_inst}{memory_note}\n\n<|user|>\n{user_input}\n\n<|assistant|>\n"

            # Special-case: answer name queries deterministically from memory
            if re.search(r"\bwhat\s+is\s+my\s+name\b", user_input, re.IGNORECASE):
                if session_state["user_name"]:
                    assistant_response = f"Your name is {session_state['user_name']}."
                else:
                    assistant_response = "I don't know your name. Say 'remember my name is <name>' to set it for this session."

                print(assistant_response)
                print()

                fb = input("Feedback (y=helpful / n=not helpful / Enter=skip): ").strip().lower()
                feedback = "positive" if fb == "y" else ("negative" if fb == "n" else "neutral")
                # Allow memory setting via feedback text
                _fb_name = update_name_from_text(fb)
                try:
                    record_feedback(feedback)
                except Exception:
                    pass
                try:
                    record_interaction(user_id="local_user", question=user_input, answer=assistant_response, feedback=feedback)
                    log_generation({
                        "latency_ms": 0.0,
                        "tokens_per_s": 0.0,
                        "gen_len": 0,
                    })
                except Exception:
                    pass
                continue
            
            # Tokenize with attention mask
            inputs = tokenizer(formatted_input, return_tensors="pt", padding=True)
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with production-optimized parameters
            print("Assistant: ", end="", flush=True)
            with torch.no_grad():
                t0 = time.time()
                end_id = tokenizer.convert_tokens_to_ids("<|end|>")
                eos_ids = [tokenizer.eos_token_id] + ([end_id] if end_id is not None else [])
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.3,  # Lower for more deterministic outputs
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
                dt = time.time() - t0
            
            # Decode without special tokens first
            full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            
            # Extract only the assistant's response (after the last <|assistant|> tag)
            if "<|assistant|>" in full_output:
                parts = full_output.split("<|assistant|>")
                assistant_response = parts[-1].strip()
            else:
                # Fallback: remove the input prompt
                assistant_response = full_output[len(formatted_input):].strip()
            
            # Stop at end token or next role marker
            for stop_marker in ["<|end|>", "<|user|>", "<|system|>"]:
                if stop_marker in assistant_response:
                    assistant_response = assistant_response.split(stop_marker)[0].strip()
            
            # Remove any remaining special tokens
            assistant_response = assistant_response.replace("<|assistant|>", "").strip()
            
            # Remove the user's question if it's being repeated
            if assistant_response.lower().startswith(user_input.lower()):
                assistant_response = assistant_response[len(user_input):].strip()
            
            # Clean up extra whitespace
            assistant_response = " ".join(assistant_response.split())
            
            print(assistant_response)
            print()

            # Feedback collection and logging
            fb = input("Feedback (y=helpful / n=not helpful / Enter=skip): ").strip().lower()
            feedback = "positive" if fb == "y" else ("negative" if fb == "n" else "neutral")
            # Allow memory update via feedback text (e.g., "remember my name is nani")
            _fb_name = update_name_from_text(fb)
            try:
                record_feedback(feedback)
            except Exception:
                pass
            try:
                gen_len = output_ids[0].shape[0] - inputs["input_ids"].shape[1]
                record_interaction(user_id="local_user", question=user_input, answer=assistant_response, feedback=feedback)
                if dt > 0:
                    log_generation({
                        "latency_ms": dt * 1000.0,
                        "tokens_per_s": gen_len / dt,
                        "gen_len": int(gen_len),
                    })
            except Exception:
                pass
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            continue

if __name__ == "__main__":
    main()
