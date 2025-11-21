#!/usr/bin/env python3
"""Train a language model with Shannon Control Unit (SCU) adaptive regularization."""

import os
import sys
import json
import csv
import argparse
import math
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scu import control, data


def setup_device_and_dtype():
    """Determine optimal device and dtype configuration."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        use_4bit = True
        print(f"Using CUDA with fp16 and 4-bit quantization")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        use_4bit = False
        print(f"Using MPS with fp32")
    else:
        device = "cpu"
        dtype = torch.float32
        use_4bit = False
        print(f"WARNING: Using CPU with fp32 - training will be slow")
    
    return device, dtype, use_4bit


def main(args):
    # Suppress tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Setup device and dtype
    device, dtype, use_4bit = setup_device_and_dtype()
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Setup quantization config if using 4-bit
    quantization_config = None
    if use_4bit and device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    
    # Load model and tokenizer
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True
    )
    # Memory-friendly defaults for finetuning
    try:
        model.config.use_cache = False
    except Exception:
        pass
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
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
    
    # Load and prepare data
    print(f"Loading training data from {args.train_data}")
    train_texts = data.load_texts_from_file(
        args.train_data if args.train_data else "data/train.txt",
        max_texts=args.max_texts
    )
    
    # Use actual block size from model config
    if hasattr(model.config, 'max_position_embeddings'):
        actual_block_size = min(model.config.max_position_embeddings, args.block_size)
    else:
        actual_block_size = args.block_size
    
    print(f"Using block size: {actual_block_size}")
    
    # Tokenize and chunk
    train_chunks = data.tokenize_and_chunk(
        train_texts, 
        tokenizer, 
        block_size=actual_block_size,
        shuffle=True,
        seed=args.seed
    )
    
    print(f"Created {len(train_chunks)} training chunks")
    
    # Calculate tokens per epoch for normalization
    if args.tokens_per_epoch_override:
        tokens_per_epoch = args.tokens_per_epoch_override
        print(f"Using OVERRIDE tokens per epoch: {tokens_per_epoch}")
    else:
        tokens_per_epoch = len(train_chunks) * actual_block_size
        print(f"Calculated tokens per epoch: {tokens_per_epoch}")
    
    # Setup optimizer (weight_decay=0 since we use ParamBPT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0  # Important: no weight decay with ParamBPT
    )
    
    # Setup scheduler
    num_training_steps = args.steps if args.steps else args.epochs * len(train_chunks) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Prepare for distributed training
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    # Initialize control variables
    lmbda = args.lambda_init
    I = 0.0
    S_hat = None
    
    # Initialize variables that might not be set if no training happens
    S_meas = 0.0
    data_bpt = 0.0
    param_bpt = 0.0
    
    # Open CSV log file
    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_path = Path(args.log_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'data_bpt', 'param_bpt', 'total_bpt', 'S', 'lambda', 'I', 'wall_time_s'])
        start_time = time.time()
    
    # Training loop
    model.train()
    global_step = 0
    
    print(f"\nStarting training for {num_training_steps} steps")
    print(f"Target S: {args.target_s:.1%}, Kp: {args.kp}, Ki: {args.ki}")
    
    data_iter = data.create_data_iterator(train_chunks, args.batch_size)
    
    for epoch in range(args.epochs if not args.steps else 1):
        if args.steps and global_step >= args.steps:
            break
            
        for batch_chunks in data_iter:
            if args.steps and global_step >= args.steps:
                break
            
            # Prepare batch
            batch_ids = torch.tensor([c['input_ids'] for c in batch_chunks])
            batch_mask = torch.tensor([c['attention_mask'] for c in batch_chunks])
            
            batch_ids = batch_ids.to(accelerator.device)
            batch_mask = batch_mask.to(accelerator.device)
            
            # Labels are same as inputs for causal LM
            labels = batch_ids.clone()
            
            # Forward pass
            outputs = model(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                labels=labels
            )
            
            # Calculate DataBPT (convert from nats to bits)
            ce_loss_nats = outputs.loss
            data_bpt = control.calculate_data_bpt(ce_loss_nats.item())
            
            # Calculate ParamBPT
            param_bpt = control.calculate_param_bpt(
                model,
                sigma=args.prior_sigma,
                tokens_per_epoch=tokens_per_epoch
            )
            
            # Calculate S ratio
            S_meas = control.calculate_s_ratio(data_bpt, param_bpt)
            
            # Update lambda with PI control
            lmbda, I, S_hat = control.update_lambda(
                lmbda, S_meas, args.target_s, I,
                Kp=args.kp, Ki=args.ki,
                deadband=args.deadband,
                lmin=args.lambda_min, lmax=args.lambda_max,
                S_hat=S_hat
            )
            
            # Calculate total loss (CE + λ * quadratic regularization)
            # Note: param_bpt already includes the quadratic term
            reg_loss = param_bpt * math.log(2) * tokens_per_epoch  # Convert back to nats
            total_loss = ce_loss_nats + lmbda * reg_loss
            
            # Backward pass
            accelerator.backward(total_loss)
            
            # Gradient accumulation
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if global_step % 10 == 0:
                print(f"Step {global_step}: DataBPT={data_bpt:.3f}, ParamBPT={param_bpt:.4f}, "
                      f"S={S_meas:.1%}, λ={lmbda:.3f}, I={I:.4f}")
            
            if csv_writer:
                total_bpt = data_bpt + param_bpt
                csv_writer.writerow([
                    global_step, 
                    f"{data_bpt:.4f}",
                    f"{param_bpt:.6f}",
                    f"{total_bpt:.6f}",
                    f"{S_meas:.4f}",
                    f"{lmbda:.4f}",
                    f"{I:.4f}",
                    f"{time.time() - start_time:.2f}"
                ])
                csv_file.flush()
            
            global_step += 1
    
    # Close CSV file
    if csv_file:
        csv_file.close()
    
    # Save adapter
    print(f"\nSaving adapter to {args.adapter_out}")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)
    
    # Save metadata
    metadata = {
        'base_model': args.base_model,
        'target_s': args.target_s,
        'final_s': S_meas,
        'final_lambda': lmbda,
        'final_data_bpt': data_bpt,
        'final_param_bpt': param_bpt,
        'kp': args.kp,
        'ki': args.ki,
        'prior_sigma': args.prior_sigma,
        'block_size': actual_block_size,
        'tokens_per_epoch': tokens_per_epoch,
        'steps': global_step,
        'seed': args.seed,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(Path(args.adapter_out) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Final: DataBPT={data_bpt:.3f}, ParamBPT={param_bpt:.4f}, S={S_meas:.1%}, λ={lmbda:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with SCU adaptive regularization")
    
    # Model args
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B",
                       help="Base model to fine-tune")
    parser.add_argument("--adapter_out", default="adapters/scu_adapter",
                       help="Output directory for adapter")
    
    # Control args
    parser.add_argument("--target_s", type=float, default=0.01,
                       help="Target S ratio (default: 0.01 = 1%%)")
    parser.add_argument("--kp", type=float, default=0.8,
                       help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.15,
                       help="Integral gain")
    parser.add_argument("--deadband", type=float, default=0.002,
                       help="Deadband for control updates")
    parser.add_argument("--lambda_init", type=float, default=1.0,
                       help="Initial lambda value")
    parser.add_argument("--lambda_min", type=float, default=1e-4,
                       help="Minimum lambda")
    parser.add_argument("--lambda_max", type=float, default=2.0,
                       help="Maximum lambda")
    
    # Training args
    parser.add_argument("--prior_sigma", type=float, default=0.01,
                       help="Prior std dev for ParamBPT")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs")
    parser.add_argument("--steps", type=int, default=None,
                       help="Number of steps (overrides epochs)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--block_size", type=int, default=1024,
                       help="Block size for chunking")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Data args
    parser.add_argument("--train_data", default="data/train.txt",
                       help="Training data file")
    parser.add_argument("--max_texts", type=int, default=None,
                       help="Maximum number of texts to load")
    
    # Logging
    parser.add_argument("--log_csv", default="logs/scu_training.csv",
                       help="CSV file for logging")
    
    # Advanced control
    parser.add_argument("--tokens_per_epoch_override", type=int, default=None,
                       help="Override calculated tokens per epoch (for complexity normalization)")
    
    # Quick start mode
    parser.add_argument("--quickstart", action="store_true",
                       help="Run quick demo with minimal data")
    
    args = parser.parse_args()
    
    # Quick start overrides
    if args.quickstart:
        args.steps = 50
        args.max_texts = 10
        args.batch_size = 2
        print("Running in quickstart mode (50 steps, minimal data)")
    
    main(args)
