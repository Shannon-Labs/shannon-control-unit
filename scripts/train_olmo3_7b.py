#!/usr/bin/env python3
"""
OLMo 3 7B - SCU Training with Unsloth (CUDA) or MLX (Mac) support

Supports:
  - RTX 3080 10GB: unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit
  - Mac M4 Max 36GB: mlx-community/Olmo-3-7B-Instruct-4bit
"""

import os
import sys
import json
import csv
import argparse
import math
import platform
from pathlib import Path
from datetime import datetime
import time

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def detect_hardware():
    """Detect available hardware and return optimal backend."""
    system = platform.system()

    if system == "Darwin":
        # Check for Apple Silicon
        try:
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                   capture_output=True, text=True)
            if 'Apple' in result.stdout:
                return "mlx", "Apple Silicon detected"
        except Exception:
            pass
        return "cpu", "Mac without Apple Silicon"

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return "cuda", f"CUDA detected: {gpu_name} ({vram_gb:.1f}GB)"
    except Exception:
        pass

    return "cpu", "No GPU detected, using CPU (slow)"


def train_cuda(args):
    """Train on CUDA with Unsloth 4-bit quantization."""
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
    from scu import control, data

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Try to use Unsloth for optimized training
    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
        print("Using Unsloth for optimized training")
    except ImportError:
        print("Unsloth not available, using standard transformers")

    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    if use_unsloth:
        # Unsloth path - handles quantization internally
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.block_size,
            dtype=torch.float16,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    else:
        # Standard transformers with BitsAndBytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        print(f"Loading model: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    model.print_trainable_parameters()

    # Load data
    print(f"Loading data from {args.train_data}")
    train_texts = data.load_texts_from_file(args.train_data, max_texts=args.max_texts)
    train_chunks = data.tokenize_and_chunk(train_texts, tokenizer,
                                           block_size=args.block_size,
                                           shuffle=True, seed=args.seed)
    print(f"Created {len(train_chunks)} training chunks")

    tokens_per_epoch = len(train_chunks) * args.block_size

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    num_training_steps = args.steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Run training loop
    return _run_training_loop(
        model, tokenizer, train_chunks, optimizer, scheduler,
        accelerator, control, data, args, tokens_per_epoch, "CUDA"
    )


def train_mlx(args):
    """Train on Apple Silicon with MLX using the production MLXTrainingEngine."""
    from shannon_control.mlx import MLXTrainingEngine, MLXTrainingConfig

    print(f"\n{'='*60}")
    print(f"OLMo 3 MLX Training with Shannon Control Unit")
    print(f"{'='*60}")
    print(f"Model: {args.base_model}")
    print(f"Target S: {args.target_s:.1%}")
    print(f"Data: {args.train_data}")
    print(f"Steps: {args.steps}")
    print(f"{'='*60}\n")

    # Create MLX training configuration
    config = MLXTrainingConfig(
        # Model and data
        base_model=args.base_model,
        train_data=args.train_data,
        val_data=None,  # No validation data for now

        # Training parameters
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        steps=args.steps,
        block_size=args.block_size,
        seed=args.seed,

        # LoRA configuration
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=32,  # OLMo 3 has 32 layers
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],

        # SCU control parameters
        target_s=args.target_s,
        kp=args.kp,
        ki=args.ki,
        deadband=args.deadband,
        lambda_init=args.lambda_init,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        prior_sigma=args.prior_sigma,
        control_frequency=10,  # Apply control every 10 steps for fine-grained control

        # Output and logging
        adapter_out=args.adapter_out,
        log_dir=str(Path(args.adapter_out).parent / "logs"),
        log_csv=args.log_csv,
        report_steps=10,
        eval_steps=0,  # No validation for now
        save_steps=args.steps,  # Save only at the end

        # MLX-specific settings
        grad_checkpoint=True,
        use_quantized_model=True,  # Using quantized model
        enable_power_monitoring=True,
    )

    # Create and run MLX training engine
    engine = MLXTrainingEngine(
        config=config,
        job_id=f"olmo3_scu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    try:
        # Run training
        adapter_path = engine.run()

        # Get final metrics
        summary = engine.scu_callback.get_metrics_summary()

        # Create comprehensive metadata
        metadata = {
            'base_model': args.base_model,
            'architecture': 'OLMo3-MLX-SCU',
            'backend': 'MLX',

            # SCU metrics
            'target_s': args.target_s,
            'final_s_ratio': summary.get('final_s_ratio', 0),
            'mean_s_ratio': summary.get('mean_s_ratio', 0),
            'final_lambda': summary.get('final_lambda', 0),
            'lambda_range': summary.get('lambda_range', [0, 0]),

            # Training metrics
            'final_loss': summary.get('final_loss', 0),
            'mean_loss': summary.get('mean_loss', 0),
            'total_steps': summary.get('total_steps', 0),

            # Configuration
            'scu_config': {
                'kp': args.kp,
                'ki': args.ki,
                'deadband': args.deadband,
                'lambda_init': args.lambda_init,
                'lambda_min': args.lambda_min,
                'lambda_max': args.lambda_max,
                'prior_sigma': args.prior_sigma,
            },
            'lora_config': {
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'target_modules': config.lora_target_modules,
            },
            'training_config': {
                'steps': args.steps,
                'batch_size': args.batch_size,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'lr': args.lr,
                'block_size': args.block_size,
                'seed': args.seed,
            },

            'adapter_path': str(adapter_path),
            'log_csv': args.log_csv,
            'timestamp': datetime.now().isoformat(),
            'success': True,
        }

        # Save metadata
        metadata_path = Path(args.adapter_out) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Final S-ratio: {metadata['final_s_ratio']:.4f} (target: {args.target_s:.4f})")
        print(f"Final lambda: {metadata['final_lambda']:.4f}")
        print(f"Final loss: {metadata['final_loss']:.4f}")
        print(f"Adapter saved to: {adapter_path}")
        print(f"Metadata saved to: {metadata_path}")
        if args.log_csv:
            print(f"Metrics logged to: {args.log_csv}")
        print(f"{'='*60}")

        return metadata

    except Exception as e:
        print(f"\n[ERROR] MLX training failed: {e}")
        print(f"Check logs in {config.log_dir} for details")

        # Return error metadata
        error_metadata = {
            'base_model': args.base_model,
            'architecture': 'OLMo3-MLX-SCU',
            'backend': 'MLX',
            'target_s': args.target_s,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e),
            'adapter_path': args.adapter_out,
        }

        # Save error metadata
        Path(args.adapter_out).mkdir(parents=True, exist_ok=True)
        with open(Path(args.adapter_out) / 'metadata.json', 'w') as f:
            json.dump(error_metadata, f, indent=2)

        raise


def _run_training_loop(model, tokenizer, train_chunks, optimizer, scheduler,
                       accelerator, control, data, args, tokens_per_epoch, backend):
    """Common training loop for CUDA/CPU backends."""
    import torch

    # Initialize control variables
    lmbda = args.lambda_init
    I = 0.0
    S_hat = None

    # Initialize metrics
    S_meas = 0.0
    data_bpt = 0.0
    param_bpt = 0.0

    # Setup logging
    csv_file = None
    csv_writer = None
    start_time = time.time()

    if args.log_csv:
        csv_path = Path(args.log_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'data_bpt', 'param_bpt', 'total_bpt', 'S', 'lambda', 'I', 'wall_time_s'])

    # Training loop
    model.train()
    global_step = 0

    print(f"\nStarting {backend} training for {args.steps} steps")
    print(f"Target S: {args.target_s:.1%}, Kp: {args.kp}, Ki: {args.ki}")

    data_iter = data.create_data_iterator(train_chunks, args.batch_size)

    for batch_chunks in data_iter:
        if global_step >= args.steps:
            break

        # Prepare batch
        batch_ids = torch.tensor([c['input_ids'] for c in batch_chunks])
        batch_mask = torch.tensor([c['attention_mask'] for c in batch_chunks])

        batch_ids = batch_ids.to(accelerator.device)
        batch_mask = batch_mask.to(accelerator.device)
        labels = batch_ids.clone()

        # Forward pass
        outputs = model(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            labels=labels
        )

        # Calculate metrics
        ce_loss_nats = outputs.loss
        data_bpt = control.calculate_data_bpt(ce_loss_nats.item())
        param_bpt = control.calculate_param_bpt(
            model, sigma=args.prior_sigma, tokens_per_epoch=tokens_per_epoch
        )
        S_meas = control.calculate_s_ratio(data_bpt, param_bpt)

        # PI control update
        lmbda, I, S_hat = control.update_lambda(
            lmbda, S_meas, args.target_s, I,
            Kp=args.kp, Ki=args.ki,
            deadband=args.deadband,
            lmin=args.lambda_min, lmax=args.lambda_max,
            S_hat=S_hat
        )

        # Total loss with regularization
        reg_loss = param_bpt * math.log(2) * tokens_per_epoch
        total_loss = ce_loss_nats + lmbda * reg_loss

        # Backward pass
        accelerator.backward(total_loss)

        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if global_step % 10 == 0:
            print(f"Step {global_step}: DataBPT={data_bpt:.3f}, ParamBPT={param_bpt:.4f}, "
                  f"S={S_meas:.1%}, lambda={lmbda:.3f}, I={I:.4f}")

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

    if csv_file:
        csv_file.close()

    # Save adapter
    print(f"\nSaving adapter to {args.adapter_out}")
    Path(args.adapter_out).mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.adapter_out)
    tokenizer.save_pretrained(args.adapter_out)

    # Save metadata
    metadata = {
        'base_model': args.base_model,
        'architecture': 'OLMo3',
        'lora_targets': ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        'target_s': args.target_s,
        'final_s': S_meas,
        'final_lambda': lmbda,
        'final_data_bpt': data_bpt,
        'final_param_bpt': param_bpt,
        'kp': args.kp,
        'ki': args.ki,
        'prior_sigma': args.prior_sigma,
        'block_size': args.block_size,
        'tokens_per_epoch': tokens_per_epoch,
        'steps': global_step,
        'backend': backend,
        'timestamp': datetime.now().isoformat()
    }

    with open(Path(args.adapter_out) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Final: DataBPT={data_bpt:.3f}, ParamBPT={param_bpt:.4f}, S={S_meas:.1%}, lambda={lmbda:.3f}")

    return metadata


def main(args):
    # Detect hardware
    backend, info = detect_hardware()
    print(f"Hardware: {info}")

    # Override backend if specified
    if args.backend:
        backend = args.backend
        print(f"Backend override: {backend}")

    # Auto-select model if not specified
    if args.base_model == "auto":
        if backend == "cuda":
            args.base_model = "unsloth/Olmo-3-7B-Instruct-unsloth-bnb-4bit"
        elif backend == "mlx":
            args.base_model = "mlx-community/Olmo-3-7B-Instruct-4bit"
        else:
            args.base_model = "allenai/OLMo-3-7B"
        print(f"Auto-selected model: {args.base_model}")

    # Route to appropriate backend
    if backend == "mlx":
        return train_mlx(args)
    else:
        return train_cuda(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OLMo 3 7B with SCU (Unsloth/MLX support)"
    )

    # Model
    parser.add_argument("--base_model", default="auto",
                       help="Model ID or 'auto' for hardware-based selection")
    parser.add_argument("--adapter_out", default="adapters/olmo3_7b_scu",
                       help="Output directory for adapter")
    parser.add_argument("--backend", choices=["cuda", "mlx", "cpu"],
                       help="Force specific backend")

    # SCU Control
    parser.add_argument("--target_s", type=float, default=0.02,
                       help="Target S ratio (default: 0.02 = 2%% for 7B)")
    parser.add_argument("--kp", type=float, default=0.8)
    parser.add_argument("--ki", type=float, default=0.15)
    parser.add_argument("--deadband", type=float, default=0.002)
    parser.add_argument("--lambda_init", type=float, default=1.0)
    parser.add_argument("--lambda_min", type=float, default=1e-4)
    parser.add_argument("--lambda_max", type=float, default=2.0)

    # Training
    parser.add_argument("--prior_sigma", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Data
    parser.add_argument("--train_data", default="data/fineweb_edu_sample.jsonl")
    parser.add_argument("--max_texts", type=int, default=None)

    # Logging
    parser.add_argument("--log_csv", default="logs/olmo3_7b_scu.csv")

    args = parser.parse_args()
    main(args)
