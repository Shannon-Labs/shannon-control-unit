#!/usr/bin/env python3
"""
CUDA-Optimized SCU Training Script for IBM Granite-4.0-H-1B

High-performance training script optimized for NVIDIA GPUs with:
- Multi-GPU support with automatic device detection
- FP16/bfloat16 mixed precision training
- Gradient checkpointing for memory efficiency
- torch.compile() optimization
- Advanced CUDA memory management
- Production-ready checkpointing and resumption

Usage:
    python scripts/train_granite_cuda.py [--fp16] [--bf16] [--batch-size 8] [--multi-gpu]
"""

import os
import sys
import argparse
import logging
import json
import time
import math
import gc
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any, List

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.granite_1b_scu_config import Granite1BSCUConfig, Granite1BModelCardConfig


def setup_logging(config: Granite1BSCUConfig) -> logging.Logger:
    """Setup CUDA-optimized logging configuration"""
    log_dir = Path(config.logging_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'cuda_optimized_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("CudaOptimized-Granite-Training")


def detect_optimal_dtype() -> str:
    """Detect optimal dtype based on GPU capabilities"""
    if not torch.cuda.is_available():
        return "float32"
    
    # Check for bfloat16 support (Ampere+ GPUs)
    if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        return "bfloat16"
    
    # Check for tensor cores (V100, T4, RTX series)
    device_props = torch.cuda.get_device_properties(0)
    if device_props.major >= 7:  # Volta and newer
        return "float16"
    
    return "float32"


def check_cuda_requirements():
    """Check if CUDA requirements are met"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This script requires NVIDIA GPU.")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA devices found: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        if memory_gb < 8:
            print(f"⚠️  Warning: GPU {i} has less than 8GB memory")
    
    return True


# Import heavy dependencies only when needed
def import_dependencies():
    """Import heavy dependencies with error handling"""
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            EarlyStoppingCallback
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset, DatasetDict
        from scu.control import update_lambda, calculate_param_bpt, calculate_data_bpt, calculate_s_ratio
        
        return {
            'AutoModelForCausalLM': AutoModelForCausalLM,
            'AutoTokenizer': AutoTokenizer,
            'TrainingArguments': TrainingArguments,
            'Trainer': Trainer,
            'DataCollatorForLanguageModeling': DataCollatorForLanguageModeling,
            'EarlyStoppingCallback': EarlyStoppingCallback,
            'LoraConfig': LoraConfig,
            'get_peft_model': get_peft_model,
            'TaskType': TaskType,
            'load_dataset': load_dataset,
            'DatasetDict': DatasetDict,
            'update_lambda': update_lambda,
            'calculate_param_bpt': calculate_param_bpt,
            'calculate_data_bpt': calculate_data_bpt,
            'calculate_s_ratio': calculate_s_ratio
        }
        
    except ImportError as e:
        raise ImportError(f"Failed to import required dependencies: {e}. Please install: pip install transformers peft datasets")


class CudaOptimizedSCUTrainer:
    """CUDA-optimized trainer with SCU integration and advanced memory management"""
    
    def __init__(self, config: Granite1BSCUConfig, model, args, train_dataset, eval_dataset=None, data_collator=None, callbacks=None):
        # Import dependencies
        deps = import_dependencies()
        
        # Create base trainer
        self.trainer = deps['Trainer'](
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # SCU-specific attributes
        self.config = config
        self.scu_initialized = False
        self.lambda_current = config.lambda_init
        self.integral_term = 0.0
        self.s_ratio_history = []
        self.lambda_history = []
        self.control_metrics = []
        self.ema_s_ratio = None
        
        # CUDA-specific attributes
        self.device_count = torch.cuda.device_count()
        self.current_device = torch.cuda.current_device()
        
        # Memory monitoring
        self.peak_memory_mb = 0
        self.memory_check_interval = 50
        self.cuda_memory_stats = []
        
        # Performance tracking
        self.step_times = []
        self.compile_time = 0
        
        # Store dependencies
        self.deps = deps
        
        # Setup logger
        self.logger = logging.getLogger("CudaOptimized-SCU-Trainer")
        
        # Initial CUDA setup
        self._setup_cuda()

    def _setup_cuda(self):
        """Setup CUDA environment and optimizations"""
        if torch.cuda.is_available():
            # Enable memory efficient attention
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Enable cuDNN benchmarking for optimal performance
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction for better allocation
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Log CUDA info
            self.logger.info(f"CUDA devices available: {self.device_count}")
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def initialize_scu(self) -> bool:
        """Initialize SCU system with CUDA optimizations"""
        try:
            # Initial memory check
            initial_memory = self._get_cuda_memory_mb()
            self.logger.info(f"Initial CUDA memory usage: {initial_memory:.1f} MB")
            
            if initial_memory > 20000:  # 20GB warning
                self.logger.warning(f"High initial CUDA memory: {initial_memory:.1f} MB")
            
            self.scu_initialized = True
            self.logger.info("SCU initialized successfully (CUDA-optimized mode)")
            self.logger.info(f"Target S ratio: {self.config.target_s_ratio:.3%}")
            self.logger.info(f"PI gains: Kp={self.config.kp}, Ki={self.config.ki}")
            self.logger.info(f"Control frequency: Every {self.config.control_frequency} steps")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SCU: {e}")
            return False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with CUDA-optimized SCU integration"""
        
        # Standard forward pass
        outputs = model(**inputs)
        ce_loss_nats = outputs.loss

        # Apply SCU control if enabled and it's time to control
        if (self.scu_initialized and 
            self.trainer.state.global_step % self.config.control_frequency == 0):
            
            try:
                # Calculate data BPT (convert from nats to bits)
                data_bpt = self.deps['calculate_data_bpt'](ce_loss_nats.item())

                # Calculate parameter BPT for LoRA weights
                param_bpt = self.deps['calculate_param_bpt'](model, sigma=self.config.prior_sigma)

                # Calculate S ratio with EMA smoothing
                s_ratio = self.deps['calculate_s_ratio'](data_bpt, param_bpt)
                
                # Apply EMA smoothing
                if self.ema_s_ratio is None:
                    self.ema_s_ratio = s_ratio
                else:
                    self.ema_s_ratio = self.ema_s_ratio * (1 - self.config.ema_alpha) + s_ratio * self.config.ema_alpha

                # Store for history
                self.s_ratio_history.append(self.ema_s_ratio)
                
                # Apply SCU control law with anti-windup
                self.lambda_current, self.integral_term, _ = self.deps['update_lambda'](
                    self.lambda_current,
                    self.ema_s_ratio,
                    self.config.target_s_ratio,
                    self.integral_term,
                    Kp=self.config.kp,
                    Ki=self.config.ki,
                    deadband=self.config.deadband,
                    lmin=self.config.lambda_min,
                    lmax=self.config.lambda_max,
                    leak=self.config.integral_leak,
                    ema_alpha=self.config.ema_alpha
                )
                
                # Apply lambda regularization
                reg_loss_nats = param_bpt * math.log(2) * self.lambda_current
                if isinstance(ce_loss_nats, torch.Tensor):
                    total_loss_nats = ce_loss_nats + torch.tensor(reg_loss_nats, device=ce_loss_nats.device, dtype=ce_loss_nats.dtype)
                else:
                    total_loss_nats = ce_loss_nats + reg_loss_nats

                # Store metrics
                metrics = {
                    'step': self.trainer.state.global_step,
                    'ce_loss': ce_loss_nats.item() if hasattr(ce_loss_nats, 'item') else ce_loss_nats,
                    'reg_loss': reg_loss_nats,
                    'total_loss': total_loss_nats.item() if hasattr(total_loss_nats, 'item') else total_loss_nats,
                    'data_bpt': data_bpt,
                    'param_bpt': param_bpt,
                    's_ratio': self.ema_s_ratio,
                    'lambda': self.lambda_current,
                    'integral_term': self.integral_term,
                    'error': self.ema_s_ratio - self.config.target_s_ratio,
                    'cuda_memory_mb': self._get_cuda_memory_mb()
                }

                self.control_metrics.append(metrics)
                self.lambda_history.append(self.lambda_current)

                # Log control action
                if self.trainer.state.global_step % self.config.logging_steps == 0:
                    self._log_control_action(metrics)

                loss = total_loss_nats

            except Exception as e:
                self.logger.error(f"SCU control error at step {self.trainer.state.global_step}: {e}")
                loss = ce_loss_nats  # Fall back to standard loss
        else:
            loss = ce_loss_nats

        # Memory check and cleanup
        if self.trainer.state.global_step % self.memory_check_interval == 0:
            self._cuda_memory_cleanup_and_check()

        return (loss, outputs) if return_outputs else loss

    def _log_control_action(self, metrics: Dict[str, Any]):
        """Log control action with CUDA memory info"""
        cuda_memory = self._get_cuda_memory_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, cuda_memory)
        
        self.logger.info(
            f"SCU Step {metrics['step']}: "
            f"S={metrics['s_ratio']:.4f} ({metrics['s_ratio']*100:.2f}%), "
            f"λ={metrics['lambda']:.4f}, "
            f"Error={metrics['error']:.4f}, "
            f"CUDA={cuda_memory:.1f}MB"
        )

    def _cuda_memory_cleanup_and_check(self):
        """Perform CUDA memory cleanup and check for issues"""
        current_memory = self._get_cuda_memory_mb()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        # Store stats
        self.cuda_memory_stats.append({
            'step': self.trainer.state.global_step,
            'memory_mb': current_memory,
            'allocated_mb': allocated,
            'reserved_mb': reserved
        })
        
        # Warning levels
        if current_memory > 30000:  # 30GB
            self.logger.warning(f"CRITICAL: CUDA memory usage {current_memory:.1f} MB - approaching limit!")
            # Aggressive cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif current_memory > 25000:  # 25GB
            self.logger.warning(f"High CUDA memory usage: {current_memory:.1f} MB")
            torch.cuda.empty_cache()

    def _get_cuda_memory_mb(self) -> float:
        """Get current CUDA memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0

    def save_model(self, output_dir: str, _internal_call: bool = False):
        """Save model with SCU metadata and CUDA state"""
        # Memory check before saving
        memory_before = self._get_cuda_memory_mb()
        self.logger.info(f"CUDA memory before save: {memory_before:.1f} MB")
        
        self.trainer.save_model(output_dir, _internal_call)
        
        # Save SCU training metrics
        if self.control_metrics:
            metrics_file = Path(output_dir) / "scu_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.control_metrics, f, indent=2)
        
        # Save S ratio and lambda history
        history_file = Path(output_dir) / "scu_history.json"
        history_data = {
            's_ratio_history': self.s_ratio_history,
            'lambda_history': self.lambda_history,
            'target_s_ratio': self.config.target_s_ratio,
            'peak_cuda_memory_mb': self.peak_memory_mb,
            'cuda_memory_stats': self.cuda_memory_stats
        }
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Save training configuration
        config_file = Path(output_dir) / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Memory check after saving
        memory_after = self._get_cuda_memory_mb()
        self.logger.info(f"CUDA memory after save: {memory_after:.1f} MB")
        
        # Cleanup after save
        gc.collect()
        torch.cuda.empty_cache()

    def train(self, resume_from_checkpoint=None):
        """Start training with SCU initialization"""
        # Override compute_loss method
        self.trainer.compute_loss = self.compute_loss
        
        # Initialize SCU
        self.initialize_scu()
        
        # Start training
        return self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def create_model_and_tokenizer_cuda(config: Granite1BSCUConfig, logger, deps):
    """Create model and tokenizer with CUDA optimizations"""
    try:
        # Determine optimal dtype
        optimal_dtype = detect_optimal_dtype()
        torch_dtype = getattr(torch, optimal_dtype)
        
        # Load tokenizer first
        tokenizer = deps['AutoTokenizer'].from_pretrained(
            config.tokenizer_name,
            use_fast=config.use_fast_tokenizer,
            trust_remote_code=config.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
        
        # Load model with CUDA optimizations
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'trust_remote_code': config.trust_remote_code,
            'low_cpu_mem_usage': True,
            'device_map': config.device_map,
        }
        
        # Add quantization config for memory efficiency
        if config.use_4bit_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs['quantization_config'] = quantization_config
        
        model = deps['AutoModelForCausalLM'].from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        logger.info(f"Base model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
        
        # Apply LoRA for memory efficiency
        if config.use_lora:
            lora_config = deps['LoraConfig'](
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
                task_type=getattr(deps['TaskType'], config.lora_task_type)
            )
            model = deps['get_peft_model'](model, lora_config)
            model.print_trainable_parameters()
        
        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Apply torch.compile for performance optimization
        if config.use_torch_compile and hasattr(torch, 'compile'):
            logger.info("Applying torch.compile optimization...")
            compile_start = time.time()
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            compile_time = time.time() - compile_start
            logger.info(f"torch.compile completed in {compile_time:.2f} seconds")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        raise


def load_and_prepare_dataset_cuda(config: Granite1BSCUConfig, tokenizer, logger, deps):
    """Load and prepare dataset with CUDA optimizations"""
    try:
        dataset = deps['load_dataset'](
            config.dataset_name,
            config.dataset_config,
            split=f'train[:{config.max_examples}]'
        )
        
        # Create validation split
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = deps['DatasetDict']({
            'train': dataset['train'],
            'validation': dataset['test']
        })
        
        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val")
        
        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=config.block_size,
                return_overflowing_tokens=False,
            )
        
        # Tokenize with optimal workers
        num_workers = min(config.preprocessing_num_workers, os.cpu_count())
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing dataset"
        )
        
        # Group texts into blocks
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            total_length = (total_length // config.block_size) * config.block_size
            result = {
                k: [t[i : i + config.block_size] for i in range(0, total_length, config.block_size)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        grouped_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts into chunks of {config.block_size}"
        )

        return grouped_dataset

    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise


def main():
    """Main CUDA-optimized training function"""
    parser = argparse.ArgumentParser(description="CUDA-optimized SCU training for Granite-4.0-H-1B")
    
    # CUDA-specific arguments
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per GPU (default: 8)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU training"
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile optimization"
    )
    parser.add_argument(
        "--4bit-quantization",
        action="store_true",
        help="Enable 4-bit quantization for memory efficiency"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps (default: 1000)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Block size for training (default: 1024)"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./granite_1b_cuda_output",
        help="Output directory"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run with minimal steps for testing"
    )
    parser.add_argument(
        "--check-requirements",
        action="store_true",
        help="Check CUDA requirements and exit"
    )

    args = parser.parse_args()

    # Check requirements if requested
    if args.check_requirements:
        check_cuda_requirements()
        return

    # Check CUDA availability
    if not check_cuda_requirements():
        return

    # Load configuration
    config = Granite1BSCUConfig()
    
    # Apply CUDA optimizations
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.max_steps = args.max_steps
    config.learning_rate = args.learning_rate
    config.block_size = args.block_size
    config.lora_r = args.lora_r
    config.fp16 = args.fp16
    config.bf16 = args.bf16
    config.use_torch_compile = args.torch_compile
    config.use_4bit_quantization = args._4bit_quantization
    
    # Test run settings
    if args.test_run:
        config.max_steps = 100
        config.save_steps = 50
        config.eval_steps = 50
        config.logging_steps = 10
        config.warmup_steps = 20
        config.output_dir += "_test"

    # Setup logging
    logger = setup_logging(config)
    logger.info("🚀 Starting CUDA-optimized SCU training for Granite-4.0-H-1B")
    logger.info(f"Target S ratio: {config.target_s_ratio:.3%}")
    logger.info(f"PI gains: Kp={config.kp}, Ki={config.ki}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Mixed precision: {'FP16' if config.fp16 else 'BF16' if config.bf16 else 'FP32'}")
    logger.info(f"Multi-GPU: {args.multi_gpu}")
    logger.info(f"Torch compile: {config.use_torch_compile}")

    # Detect optimal dtype
    optimal_dtype = detect_optimal_dtype()
    logger.info(f"Detected optimal dtype: {optimal_dtype}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / "scu_config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Import dependencies
    try:
        deps = import_dependencies()
        logger.info("✅ All dependencies loaded successfully")
    except ImportError as e:
        logger.error(str(e))
        return

    # Create model and tokenizer
    logger.info("Loading Granite-4.0-H-1B model and tokenizer...")
    try:
        model, tokenizer = create_model_and_tokenizer_cuda(config, logger, deps)
        logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return

    # Load dataset
    logger.info("Loading and preparing dataset...")
    try:
        dataset = load_and_prepare_dataset_cuda(config, tokenizer, logger, deps)
        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val")
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return

    # Data collator
    data_collator = deps['DataCollatorForLanguageModeling'](
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if (config.fp16 or config.bf16) else None
    )

    # Training arguments - CUDA optimized
    training_args = deps['TrainingArguments'](
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.0,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        logging_dir=config.logging_dir,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy="steps" if config.run_validation else "no",
        eval_steps=config.eval_steps if config.run_validation else None,
        prediction_loss_only=True,
        fp16=config.fp16,
        bf16=config.bf16,
        dataloader_num_workers=min(4, os.cpu_count()),  # Optimized for CUDA
        dataloader_pin_memory=True,  # Enable for CUDA
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=config.save_total_limit,
        save_safetensors=config.save_safetensors,
        push_to_hub=config.push_to_hub,
        hub_token=config.hub_token,
        hub_model_id=config.hub_model_id,
        report_to=config.report_to,
        logging_first_step=config.logging_first_step,
        # CUDA optimizations
        gradient_checkpointing=config.gradient_checkpointing,
        tf32=True if torch.cuda.is_available() else False,  # Enable TF32 on Ampere
        # Multi-GPU support
        local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        deepspeed=None,
        # Memory optimization
        remove_unused_columns=False,
    )

    # Create CUDA-optimized trainer
    trainer = CudaOptimizedSCUTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if config.run_validation else None,
        data_collator=data_collator,
        callbacks=[deps['EarlyStoppingCallback'](early_stopping_patience=3)] if config.run_validation else None
    )

    # Start training
    logger.info("🔬 Starting CUDA-optimized SCU training...")
    start_time = time.time()
    
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        training_time = time.time() - start_time
        logger.info(f"SCU training completed in {training_time:.2f} seconds")

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {config.output_dir}")

        # Final summary
        final_memory = trainer._get_cuda_memory_mb()
        logger.info("🎯 Final Training Summary:")
        logger.info(f"  Total training time: {training_time:.2f} seconds")
        logger.info(f"  Peak CUDA memory usage: {trainer.peak_memory_mb:.1f} MB")
        logger.info(f"  Final CUDA memory usage: {final_memory:.1f} MB")
        logger.info(f"  Control actions applied: {len(trainer.control_metrics)}")
        
        if trainer.s_ratio_history:
            avg_s_ratio = sum(trainer.s_ratio_history) / len(trainer.s_ratio_history)
            logger.info(f"  Average S ratio: {avg_s_ratio:.4f} (target: {config.target_s_ratio:.4f})")

        logger.info("🚀 CUDA-optimized SCU training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Final memory state on error
        if torch.cuda.is_available():
            error_memory = torch.cuda.memory_allocated() / 1024**2
            logger.error(f"CUDA memory at error: {error_memory:.1f} MB")
        raise


if __name__ == "__main__":
    main()