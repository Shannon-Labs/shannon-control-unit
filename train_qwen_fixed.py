#!/usr/bin/env python3
"""
Fixed Qwen3 training with SCU - simplified model loading
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import csv
import time
from datetime import datetime
import json
from scu import control, data

def train_qwen_scu():
    """Train Qwen3 with SCU using simplified setup"""
    
    print("=" * 80)
    print("QWEN3 + SCU v1.0 TRAINING")
    print("=" * 80)
    
    # Setup
    base_model = "./Qwen3-1.7B-Base"
    adapter_out = "adapters/qwen_scu_fixed"
    train_data = "data/train.txt"
    max_texts = 500
    steps = 50
    batch_size = 1
    
    print(f"Model: {base_model}")
    print(f"Adapter: {adapter_out}")
    print(f"Data: {train_data} (max {max_texts} texts)")
    print(f"Steps: {steps}, Batch size: {batch_size}")
    print()
    
    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        print("Using MPS")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU")
    
    # Load model - simplified, no device_map issues
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(device)
    
    # Memory optimizations
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except:
        pass
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup LoRA
    print("Setting up LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        inference_mode=False
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load data
    print("\nLoading data...")
    train_texts = data.load_texts_from_file(train_data, max_texts=max_texts)
    
    # Use smaller block size for MPS memory
    block_size = 512 if device == "mps" else 1024
    
    train_chunks = data.tokenize_and_chunk(
        train_texts, tokenizer, block_size=block_size, shuffle=True, seed=42
    )
    
    print(f"Created {len(train_chunks)} training chunks")
    tokens_per_epoch = len(train_chunks) * block_size
    print(f"Tokens per epoch: {tokens_per_epoch:,}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.0)
    
    # Initialize SCU controller
    lmbda = 1.0
    I = 0.0
    S_hat = None
    target_s = 0.01
    
    # Setup logging
    Path(adapter_out).mkdir(parents=True, exist_ok=True)
    log_path = Path(adapter_out) / "training_log.csv"
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'data_bpt', 'param_bpt', 'total_bpt', 'S', 'lambda', 'I'])
    
    # Training loop
    print("\nStarting training...")
    print(f"Target S: {target_s:.1%}, Kp: 0.8, Ki: 0.15")
    print("-" * 80)
    
    model.train()
    data_iter = data.create_data_iterator(train_chunks, batch_size)
    
    start_time = time.time()
    
    for step, batch_chunks in enumerate(data_iter):
        if step >= steps:
            break
        
        # Prepare batch
        batch_ids = torch.tensor([c['input_ids'] for c in batch_chunks]).to(device)
        batch_mask = torch.tensor([c['attention_mask'] for c in batch_chunks]).to(device)
        labels = batch_ids.clone()
        
        # Forward pass
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=labels)
        
        # Calculate metrics
        ce_loss_nats = outputs.loss
        data_bpt = control.calculate_data_bpt(ce_loss_nats.item())
        
        param_bpt = control.calculate_param_bpt(
            model, sigma=0.01, tokens_per_epoch=tokens_per_epoch
        )
        
        S_meas = control.calculate_s_ratio(data_bpt, param_bpt)
        
        # Update SCU controller
        lmbda, I, S_hat = control.update_lambda(
            lmbda, S_meas, target_s, I,
            Kp=0.8, Ki=0.15, deadband=0.002,
            lmin=1e-4, lmax=2.0, S_hat=S_hat
        )
        
        # Calculate total loss with SCU regularization
        reg_loss = param_bpt * math.log(2) * tokens_per_epoch
        total_loss = ce_loss_nats + lmbda * reg_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging
        if step % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:3d} | DataBPT={data_bpt:.3f} | ParamBPT={param_bpt:.4f} | "
                  f"S={S_meas:.1%} | λ={lmbda:.3f} | I={I:.4f} | {elapsed:.1f}s")
        
        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            total_bpt = data_bpt + param_bpt
            writer.writerow([
                step, 
                f"{data_bpt:.4f}",
                f"{param_bpt:.6f}",
                f"{total_bpt:.6f}",
                f"{S_meas:.4f}",
                f"{lmbda:.4f}",
                f"{I:.4f}"
            ])
    
    print("-" * 80)
    
    # Save adapter
    print(f"\nSaving adapter to {adapter_out}")
    model.save_pretrained(adapter_out)
    tokenizer.save_pretrained(adapter_out)
    
    # Save metadata
    metadata = {
        'base_model': base_model,
        'target_s': target_s,
        'final_s': S_meas,
        'final_lambda': lmbda,
        'final_data_bpt': data_bpt,
        'final_param_bpt': param_bpt,
        'kp': 0.8,
        'ki': 0.15,
        'prior_sigma': 0.01,
        'block_size': block_size,
        'tokens_per_epoch': tokens_per_epoch,
        'steps': steps,
        'seed': 42,
        'timestamp': datetime.now().isoformat(),
        'device': device
    }
    
    with open(Path(adapter_out) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Final: DataBPT={data_bpt:.3f}, ParamBPT={param_bpt:.4f}, S={S_meas:.1%}, λ={lmbda:.3f}")
    
    return True

if __name__ == "__main__":
    import math
    try:
        train_qwen_scu()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)