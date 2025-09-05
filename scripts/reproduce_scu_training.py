#!/usr/bin/env python3
"""
Reproduce the exact SCU training that achieved 15.6% perplexity improvement.

Based on the validated configuration from models/scu_fixed_sigma_20250903_222442
that achieved:
- Base Model: 3.920 BPT | Perplexity: 15.14
- SCU-Trained: 3.676 BPT | Perplexity: 12.78
- Improvement: 15.6% lower perplexity

This script uses the exact parameters from run_fixed_with_proper_sigma.sh
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import math
import json
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Add parent dir for imports
sys.path.append(str(Path(__file__).parent))
from scu import control

def load_training_data(tokenizer, block_size=1024, num_samples=1080):
    """Load the same training data used in original training."""
    print("Loading training data...")
    
    # Use the same dataset as original
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # Tokenize and prepare samples
    samples = []
    text_buffer = ""
    
    for item in dataset:
        text_buffer += item["text"] + " "
        
        # When buffer is large enough, tokenize and split into blocks
        if len(text_buffer) > block_size * 10:
            tokens = tokenizer(text_buffer, truncation=False)["input_ids"]
            
            # Split into blocks
            for i in range(0, len(tokens) - block_size, block_size):
                block = tokens[i:i + block_size]
                samples.append({
                    "input_ids": torch.tensor(block),
                    "labels": torch.tensor(block)
                })
                
                if len(samples) >= num_samples:
                    return samples
            
            text_buffer = ""
    
    return samples

def reproduce_scu_training():
    """Reproduce the exact training that achieved 15.6% improvement."""
    
    print("="*70)
    print("REPRODUCING SCU TRAINING - Exact Configuration")
    print("="*70)
    print("Target: 15.6% perplexity reduction (3.920 → 3.676 BPT)")
    print("Configuration from: models/scu_fixed_sigma_20250903_222442")
    print("="*70)
    
    # EXACT parameters from the successful run
    config = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "max_steps": 270,  # Exact step count
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-4,
        "block_size": 1024,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "prior_sigma": 0.1,  # Critical parameter!
        "target_S": 0.01,  # 1% target
        "Kp": 0.8,  # PI controller proportional gain
        "Ki": 0.15,  # PI controller integral gain
        "deadband": 0.002,  # ±0.2pp deadband
        "lambda_min": 0.001,
        "lambda_max": 10.0,
        "warmup_steps": 27,  # 10% warmup
    }
    
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    # Load model
    print(f"\nLoading base model: {config['model_name']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        use_cache=False,  # Critical for training!
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing (memory optimization)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure LoRA with EXACT parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    
    # Apply LoRA
    print("\nApplying LoRA adapter...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load training data
    train_samples = load_training_data(
        tokenizer, 
        block_size=config["block_size"],
        num_samples=config["max_steps"] * config["gradient_accumulation_steps"]
    )
    
    print(f"Loaded {len(train_samples)} training samples")
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_samples,
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    # Optimizer - AdamW with exact parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0  # No weight decay in original
    )
    
    # Learning rate schedule with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["max_steps"]
    )
    
    # Initialize SCU control variables
    lambda_val = 0.1  # Initial lambda
    integral = 0.0
    S_hat = None
    
    # Training metrics
    metrics_history = []
    
    print("\n" + "="*70)
    print("STARTING TRAINING - Targeting 15.6% Perplexity Reduction")
    print("="*70)
    
    model.train()
    step = 0
    accumulation_counter = 0
    optimizer.zero_grad()
    
    # Training loop
    for epoch in range(10):  # Multiple epochs if needed
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            if step >= config["max_steps"]:
                break
            
            # Move batch to device
            input_ids = batch["input_ids"].unsqueeze(0).to(device)  # Add batch dim
            labels = batch["labels"].unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            ce_loss = outputs.loss
            
            # Calculate Parameter BPT (critical calculation!)
            param_sum = 0
            param_count = 0
            for name, param in model.named_parameters():
                if param.requires_grad and "lora" in name.lower():
                    param_sum += torch.sum(param ** 2).item()
                    param_count += param.numel()
            
            # Prior term: -log P(θ) = (1/2σ²) * ||θ||²
            param_bpt = (param_sum / (2 * config["prior_sigma"]**2)) / (
                config["block_size"] * config["batch_size"] * math.log(2)
            )
            
            # Data BPT
            data_bpt = ce_loss.item() / math.log(2)
            
            # MDL loss with current lambda
            mdl_loss = ce_loss + lambda_val * param_bpt
            
            # Scale loss for gradient accumulation
            scaled_loss = mdl_loss / config["gradient_accumulation_steps"]
            scaled_loss.backward()
            
            accumulation_counter += 1
            
            # Update weights after gradient accumulation
            if accumulation_counter % config["gradient_accumulation_steps"] == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Calculate S ratio
                S_measured = param_bpt / (data_bpt + param_bpt)
                
                # Update lambda using SCU control
                lambda_val, integral, S_hat = control.update_lambda(
                    lambda_val,
                    S_measured,
                    config["target_S"],
                    integral,
                    Kp=config["Kp"],
                    Ki=config["Ki"],
                    deadband=config["deadband"],
                    lmin=config["lambda_min"],
                    lmax=config["lambda_max"],
                    S_hat=S_hat,
                    ema_alpha=0.1,
                    leak=0.995
                )
                
                step += 1
                
                # Log progress
                if step % 10 == 0:
                    total_bpt = data_bpt + lambda_val * param_bpt
                    perplexity = math.exp(data_bpt * math.log(2))
                    
                    print(f"\nStep {step:3d}/{config['max_steps']} | "
                          f"Data BPT: {data_bpt:.3f} | "
                          f"Perplexity: {perplexity:.2f} | "
                          f"S: {S_measured:.1%} (target: {config['target_S']:.1%}) | "
                          f"λ: {lambda_val:.3f}")
                    
                    metrics_history.append({
                        "step": step,
                        "data_bpt": data_bpt,
                        "param_bpt": param_bpt,
                        "total_bpt": total_bpt,
                        "perplexity": perplexity,
                        "S": S_measured,
                        "lambda": lambda_val,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })
            
            if step >= config["max_steps"]:
                break
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    # Final metrics
    final_metrics = metrics_history[-1] if metrics_history else {}
    print(f"\nFinal Results:")
    print(f"  Data BPT: {final_metrics.get('data_bpt', 0):.3f}")
    print(f"  Perplexity: {final_metrics.get('perplexity', 0):.2f}")
    print(f"  S: {final_metrics.get('S', 0):.1%}")
    print(f"  Lambda: {final_metrics.get('lambda', 0):.3f}")
    
    # Check if we achieved the target
    target_bpt = 3.676
    if final_metrics.get('data_bpt', 999) <= target_bpt:
        print(f"\n✅ SUCCESS! Achieved target BPT ≤ {target_bpt}")
        print(f"   Matches validated result from original training!")
    else:
        print(f"\n⚠️  BPT higher than target {target_bpt}")
        print(f"   May need more steps or tuning")
    
    # Save model
    output_dir = Path("reproduced-scu-model")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump({
            "configuration": config,
            "final_metrics": final_metrics,
            "history": metrics_history,
            "target_achievement": {
                "target_bpt": target_bpt,
                "achieved_bpt": final_metrics.get('data_bpt', 0),
                "success": final_metrics.get('data_bpt', 999) <= target_bpt
            }
        }, f, indent=2)
    
    print(f"\n✅ Model and metrics saved to {output_dir}")
    print("\nTo test the model:")
    print("  python test_mdl_models.py")
    
    return model, tokenizer, metrics_history

if __name__ == "__main__":
    model, tokenizer, metrics = reproduce_scu_training()