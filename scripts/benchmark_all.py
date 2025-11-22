#!/usr/bin/env python3
"""
Benchmark VibeThinker Models: Base vs Baseline vs V3 vs V4
Evaluates Perplexity (DataBPT) on held-out validation data.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import math
import sys
import os
import gc

# Configuration
BASE_MODEL_ID = "models/VibeThinker-1.5B"
VAL_DATA_PATH = "data/val.txt"
ADAPTERS = {
    "Baseline (Std Finetune)": "adapters/vibethinker_1.5b_baseline",
    "V3 (Scientific SCU)": "adapters/vibethinker_1.5b_v3",
    "V4 (Adaptive SCU)": "adapters/vibethinker_1.5b_v4"
}

def load_val_data(path, tokenizer, block_size=1024, max_samples=100):
    if not os.path.exists(path):
        print(f"Error: Validation data not found at {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Use subset for speed if needed
    lines = lines[:max_samples]
    
    encodings = tokenizer("\n\n".join(lines), return_tensors="pt")
    
    # Chunk
    input_ids = encodings.input_ids
    total_length = input_ids.size(1)
    chunks = []
    for i in range(0, total_length, block_size):
        if i + block_size <= total_length:
            chunks.append(input_ids[:, i:i+block_size])
    
    return chunks

def evaluate_model(model, chunks, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.to(device)
            outputs = model(input_ids=chunk, labels=chunk)
            # Loss is mean NLL per token
            nll = outputs.loss.item() * chunk.size(1)
            total_nll += nll
            total_tokens += chunk.size(1)
    
    if total_tokens == 0:
        return float('inf'), float('inf')
        
    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    bpt = avg_nll / math.log(2)  # Bits per token
    
    return perplexity, bpt

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading Tokenizer from {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    print(f"Loading Validation Data from {VAL_DATA_PATH}")
    chunks = load_val_data(VAL_DATA_PATH, tokenizer)
    print(f"Validation chunks: {len(chunks)}")
    
    results = []
    
    # 1. Evaluate Base Model
    print("\n--- Evaluating Base Model ---")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device != "cpu" else None
    )
    
    # Check if using MPS (for Mac) to move manually if needed, but device_map="auto" might fail on MPS 
    # usually explicit to(device) is safer for simple scripts
    if device == "mps" or device == "cpu":
        model.to(device)
        
    ppl, bpt = evaluate_model(model, chunks, device)
    print(f"Base Model: PPL={ppl:.2f}, BPT={bpt:.4f}")
    results.append(("Base Model", ppl, bpt))
    
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    # 2. Evaluate Adapters
    for name, adapter_path in ADAPTERS.items():
        print(f"\n--- Evaluating {name} ---")
        if not os.path.exists(adapter_path):
            print(f"Skipping {name} (Adapter not found at {adapter_path})")
            continue
            
        # Load base again
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device != "cpu" else None
        )
        if device == "mps" or device == "cpu":
            model.to(device)
            
        # Load adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        
        ppl, bpt = evaluate_model(model, chunks, device)
        print(f"{name}: PPL={ppl:.2f}, BPT={bpt:.4f}")
        results.append((name, ppl, bpt))
        
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Print Summary
    print("\n" + "="*60)
    print(f"{ 'Model':<25} | {'Perplexity':<12} | {'Bits/Token':<12}")
    print("-" * 60)
    for name, ppl, bpt in results:
        print(f"{name:<25} | {ppl:<12.2f} | {bpt:<12.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
