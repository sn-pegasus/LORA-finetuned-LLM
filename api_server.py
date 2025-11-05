#!/usr/bin/env python3
"""
FastAPI server for the offline chatbot
Provides REST API endpoints for the Next.js frontend
"""
import os
import sys
import time
import torch
import re
import json
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from continuous_learning.versioning import get_current_adapter_path
    from continuous_learning.ingest import record_interaction
    from continuous_learning.feedback import record_feedback
    from continuous_learning.monitoring import log_generation
except Exception:
    get_current_adapter_path = lambda default=None: default or "./llama8b-production-v1"
    def record_interaction(*args, **kwargs):
        return {}
    def record_feedback(*args, **kwargs):
        return None
    def log_generation(*args, **kwargs):
        return None

app = FastAPI(title="Offline Chatbot API")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
model = None
tokenizer = None
session_states: Dict[str, Dict] = {}  # Store session states per user_id

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    user_id: Optional[str] = "default"

class FeedbackRequest(BaseModel):
    message_id: Optional[str] = None
    feedback: str  # "positive", "negative", or "neutral"
    user_id: Optional[str] = "default"

def load_model():
    """Load the model and tokenizer (called once at startup)"""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        return
    
    print("[*] Loading model (this takes about 60-90 seconds)...")
    
    adapter_path = get_current_adapter_path("./llama8b-production-v1/checkpoint-55")
    base_model_id = "meta-llama/Llama-3.2-3B"
    
    try:
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                bm = cfg.get("base_model_name_or_path")
                if isinstance(bm, str) and bm.strip():
                    base_model_id = bm.strip()
    except Exception:
        pass
    
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
    
    base_model.resize_token_embeddings(len(tokenizer))
    print("[*] Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[OK] Model loaded successfully!")

def update_name_from_text(text: str, user_id: str):
    """Update user name from conversation text"""
    try:
        if user_id not in session_states:
            session_states[user_id] = {"user_name": None}
        
        t = text or ""
        if re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
            session_states[user_id]["user_name"] = None
            return None
        if not re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
            m = re.search(r"\b(?:remember|remeber|rememeber)?\s*my\s+name\s+(?:is|as)\s+([A-Za-z][A-Za-z .'-]*)", t, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                session_states[user_id]["user_name"] = name
                return name
    except Exception:
        pass
    return None

def memory_command_reply(text: str, user_id: str):
    """Handle memory commands and return reply if applicable"""
    try:
        if user_id not in session_states:
            session_states[user_id] = {"user_name": None}
        
        t = text or ""
        if re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
            session_states[user_id]["user_name"] = None
            return "Okay, I forgot your name for this session."
        m = re.search(r"\b(?:remember|remeber|rememeber)?\s*my\s+name\s+(?:is|as)\s+([A-Za-z][A-Za-z .'-]*)", t, re.IGNORECASE)
        if m and not re.search(r"\b(?:forget|clear|reset|remove)\s+my\s+name\b", t, re.IGNORECASE):
            name = m.group(1).strip()
            session_states[user_id]["user_name"] = name
            return f"Got it. I'll remember your name is {name} for this session."
    except Exception:
        pass
    return None

def generate_response(user_input: str, user_id: str = "default") -> str:
    """Generate a response from the model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Initialize session state if needed
    if user_id not in session_states:
        session_states[user_id] = {"user_name": None}
    
    # Handle memory commands
    mem_reply = memory_command_reply(user_input, user_id)
    if mem_reply is not None:
        return mem_reply
    
    # Update name silently
    _ = update_name_from_text(user_input, user_id)
    
    # Format prompt
    system_inst = (
        "You are a pharmaceutical batch record assistant. Provide precise, factual answers based only on the batch data. "
        "Keep responses concise and accurate. Never invent personal identity or names. "
        "If asked about the user's name, only answer using the session memory. "
        "If the name is not set, state that you don't know."
    )
    memory_note = f" Known user name: {session_states[user_id]['user_name']}" if session_states[user_id]['user_name'] else ""
    formatted_input = f"<|system|>\n{system_inst}{memory_note}\n\n<|user|>\n{user_input}\n\n<|assistant|>\n"
    
    # Handle name queries
    if re.search(r"\bwhat\s+is\s+my\s+name\b", user_input, re.IGNORECASE):
        if session_states[user_id]["user_name"]:
            return f"Your name is {session_states[user_id]['user_name']}."
        else:
            return "I don't know your name. Say 'remember my name is <name>' to set it for this session."
    
    # Tokenize
    inputs = tokenizer(formatted_input, return_tensors="pt", padding=True)
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        end_id = tokenizer.convert_tokens_to_ids("<|end|>")
        eos_ids = [tokenizer.eos_token_id] + ([end_id] if end_id is not None else [])
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    
    # Decode response
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    if "<|assistant|>" in full_output:
        parts = full_output.split("<|assistant|>")
        assistant_response = parts[-1].strip()
    else:
        assistant_response = full_output[len(formatted_input):].strip()
    
    # Clean up
    for stop_marker in ["<|end|>", "<|user|>", "<|system|>"]:
        if stop_marker in assistant_response:
            assistant_response = assistant_response.split(stop_marker)[0].strip()
    
    assistant_response = assistant_response.replace("<|assistant|>", "").strip()
    
    if assistant_response.lower().startswith(user_input.lower()):
        assistant_response = assistant_response[len(user_input):].strip()
    
    assistant_response = " ".join(assistant_response.split())
    
    return assistant_response

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Offline Chatbot API", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        start_time = time.time()
        response_text = generate_response(request.message, request.user_id)
        latency_ms = (time.time() - start_time) * 1000
        
        # Log interaction
        try:
            record_interaction(
                user_id=request.user_id,
                question=request.message,
                answer=response_text,
                feedback="neutral"
            )
            log_generation({
                "latency_ms": latency_ms,
                "tokens_per_s": len(response_text.split()) / (latency_ms / 1000) if latency_ms > 0 else 0,
                "gen_len": len(response_text.split()),
            })
        except Exception:
            pass
        
        return {
            "response": response_text,
            "latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint (simulated - returns full response in chunks)"""
    try:
        response_text = generate_response(request.message, request.user_id)
        
        # Simulate streaming by chunking the response
        def generate():
            words = response_text.split()
            chunk = ""
            for i, word in enumerate(words):
                chunk += word + " "
                if (i + 1) % 3 == 0 or i == len(words) - 1:
                    yield f"data: {json.dumps({'content': chunk.strip(), 'done': False})}\n\n"
                    chunk = ""
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a message"""
    try:
        record_feedback(request.feedback)
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

