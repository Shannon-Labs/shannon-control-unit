#!/usr/bin/env python3
"""Enhanced SCU training script to push for better performance."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import math
import json
from pathlib import Path
import sys

# Add parent dir for imports
sys.path.append(str(Path(__file__).parent))
from scu import control, data

def train_enhanced_scu(
    base_model_name="meta-llama/Llama-3.2-1B",
    num_steps=1000,  # Double the original steps
    S_target=0.01,   # Target 1% like validated model
    sigma=0.005,     # Tighter prior (was 0.01)
    learning_rate=2e-4,
    batch_size=1,
    gradient_accumulation_steps=4,
    block_size=1024,
    output_dir="enhanced-scu"
):
    """Train with enhanced configuration to maximize improvements."""
    
    print("="*60)
    print("ENHANCED SCU TRAINING")
    print("="*60)
    print(f"Model: {base_model_name}")
    print(f"Steps: {num_steps}")
    print(f"S_target: {S_target:.1%}")
    print(f"Sigma: {sigma}")
    print(f"Learning rate: {learning_rate}")
    print("="*60)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        use_cache=False  # Disable for training
    )
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA with optimal settings
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # Increased rank for more capacity (was 16)
        lora_alpha=32,  # Match r for scaling
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    
    # Apply LoRA
    print("Applying LoRA adapter...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load training data
    print("\nLoading training data...")
    train_dataset = data.load_dataset(
        dataset_name="codeparrot/github-code",
        tokenizer=tokenizer,
        block_size=block_size,
        num_samples=num_steps * batch_size * gradient_accumulation_steps
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Optimizer with weight decay for better regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # Add weight decay
        betas=(0.9, 0.999)
    )
    
    # Learning rate schedule
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_steps // 10,  # 10% warmup
        num_training_steps=num_steps
    )
    
    # Initialize SCU control variables
    lambda_val = 0.1  # Start with reasonable lambda
    integral = 0.0
    S_hat = None
    
    # Enhanced PI controller parameters for tighter control
    Kp = 1.2  # Increased proportional gain (was 0.8)
    Ki = 0.25  # Increased integral gain (was 0.15)
    deadband = 0.001  # Tighter deadband (was 0.002)
    
    print("\nStarting enhanced training...")
    print(f"PI Controller: Kp={Kp}, Ki={Ki}, deadband={deadband}")
    
    model.train()
    metrics = []
    
    for step, batch in enumerate(train_dataloader):
        if step >= num_steps:
            break
        
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        ce_loss = outputs.loss
        
        # Calculate Parameter BPT with tighter prior
        param_bpt = control.calculate_param_bpt(
            model,
            sigma=sigma,
            tokens_per_epoch=block_size * batch_size * gradient_accumulation_steps * num_steps
        )
        
        # Calculate Data BPT
        data_bpt = ce_loss.item() / math.log(2)
        
        # MDL loss with current lambda
        mdl_loss = ce_loss + lambda_val * param_bpt
        
        # Backward pass
        mdl_loss.backward()
        
        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update lambda with enhanced control
            S_measured = param_bpt / (data_bpt + param_bpt)
            lambda_val, integral, S_hat = control.update_lambda(
                lambda_val,
                S_measured,
                S_target,
                integral,
                Kp=Kp,
                Ki=Ki,
                deadband=deadband,
                lmin=0.001,
                lmax=20.0,  # Allow higher lambda if needed
                S_hat=S_hat,
                leak=0.998  # Slower leak for better integral memory
            )
        
        # Log progress
        if step % 50 == 0:
            print(f"Step {step:4d} | Data BPT: {data_bpt:.3f} | "
                  f"S: {S_measured:.1%} | λ: {lambda_val:.3f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
            
            metrics.append({
                "step": step,
                "data_bpt": data_bpt,
                "param_bpt": param_bpt,
                "S": S_measured,
                "lambda": lambda_val,
                "learning_rate": scheduler.get_last_lr()[0]
            })
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final Data BPT: {data_bpt:.3f}")
    print(f"Final S: {S_measured:.1%} (target was {S_target:.1%})")
    print(f"Final λ: {lambda_val:.3f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save metrics
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump({
            "configuration": {
                "num_steps": num_steps,
                "S_target": S_target,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "Kp": Kp,
                "Ki": Ki
            },
            "final_metrics": {
                "data_bpt": data_bpt,
                "param_bpt": param_bpt,
                "S": S_measured,
                "lambda": lambda_val
            },
            "history": metrics
        }, f, indent=2)
    
    print(f"✅ Model and metrics saved to {output_path}")
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced SCU Training")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--s-target", type=float, default=0.01, help="Target S ratio")
    parser.add_argument("--sigma", type=float, default=0.005, help="Prior sigma")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Base model")
    parser.add_argument("--output", type=str, default="enhanced-scu", help="Output directory")
    
    args = parser.parse_args()
    
    train_enhanced_scu(
        base_model_name=args.model,
        num_steps=args.steps,
        S_target=args.s_target,
        sigma=args.sigma,
        learning_rate=args.lr,
        output_dir=args.output
    )