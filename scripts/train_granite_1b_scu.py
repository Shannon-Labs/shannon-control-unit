#!/usr/bin/env python3
"""
Memory-Efficient SCU Training Script for IBM Granite-4.0-H-1B

This script implements SCU training with aggressive memory optimization
for Apple Silicon and CPU-only environments. Prevents memory explosions
through careful resource management.

Usage:
    python scripts/train_granite_1b_scu.py [--test-run] [--memory-safe]
"""

import os
import sys
import argparse
import logging
import json
import time
import math
import gc
import psutil
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any

import torch
import numpy as np
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.granite_1b_scu_config import Granite1BSCUConfig, Granite1BModelCardConfig
from scu.control import update_lambda  # Use original SCU v1.0 for stability


class MemoryEfficientSCUTrainer(Trainer):
    """Trainer with memory-efficient SCU integration and explosion prevention"""

    def __init__(self, config: Granite1BSCUConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.scu_initialized = False
        self.lambda_current = config.lambda_init
        self.integral_term = 0.0
        self.s_ratio_history = []
        self.lambda_history = []
        self.control_metrics = []
        
        # Memory monitoring
        self.peak_memory_mb = 0
        self.memory_check_interval = 50
        
        # Setup logger
        self.logger = logging.getLogger("MemoryEfficient-SCU-Trainer")
        
        # Force garbage collection initially
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def initialize_scu(self) -> bool:
        """Initialize SCU system with memory checks"""
        try:
            # Initial memory check
            initial_memory = self._get_memory_usage_mb()
            self.logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
            
            if initial_memory > 30000:  # 30GB warning
                self.logger.warning(f"High initial memory: {initial_memory:.1f} MB")
            
            self.scu_initialized = True
            self.logger.info("SCU initialized successfully (memory-efficient mode)")
            self.logger.info(f"Target S ratio: {self.config.target_s_ratio:.3%}")
            self.logger.info(f"PI gains: Kp={self.config.kp}, Ki={self.config.ki}")
            self.logger.info(f"Control frequency: Every {self.config.control_frequency} steps")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SCU: {e}")
            return False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with memory-efficient SCU integration"""
        
        # Standard forward pass
        outputs = model(**inputs)
        ce_loss_nats = outputs.loss

        # Apply SCU control if enabled and it's time to control
        if (self.scu_initialized and 
            self.state.global_step % self.config.control_frequency == 0):
            
            try:
                # Calculate data BPT (convert from nats to bits)
                data_bpt = ce_loss_nats / math.log(2)

                # Calculate parameter BPT for LoRA weights (memory efficient)
                param_bpt = self._calculate_param_bpt_memory_efficient(model)

                # Calculate S ratio
                total_bpt = data_bpt + param_bpt
                s_ratio = (param_bpt / total_bpt).item() if total_bpt > 0 else 0.0

                # Store for history
                self.s_ratio_history.append(s_ratio)
                
                # Apply SCU control law (memory efficient version)
                self.lambda_current, self.integral_term, error = update_lambda(
                    self.lambda_current,
                    s_ratio,
                    self.config.target_s_ratio,
                    self.integral_term,
                    Kp=self.config.kp,
                    Ki=self.config.ki
                )
                
                # Clamp lambda to safe bounds
                self.lambda_current = max(self.config.lambda_min, 
                                        min(self.config.lambda_max, self.lambda_current))
                
                # Clamp integral term to prevent windup
                self.integral_term = max(-1.0, min(1.0, self.integral_term))

                # Apply lambda regularization (memory efficient)
                reg_loss_nats = param_bpt * math.log(2) * self.lambda_current
                total_loss_nats = ce_loss_nats + reg_loss_nats

                # Store metrics
                metrics = {
                    'step': self.state.global_step,
                    'ce_loss': ce_loss_nats.item(),
                    'reg_loss': reg_loss_nats.item(),
                    'total_loss': total_loss_nats.item(),
                    'data_bpt': data_bpt.item(),
                    'param_bpt': param_bpt.item(),
                    's_ratio': s_ratio,
                    'lambda': self.lambda_current,
                    'integral_term': self.integral_term,
                    'error': error
                }

                self.control_metrics.append(metrics)
                self.lambda_history.append(self.lambda_current)

                # Log control action
                if self.state.global_step % self.config.logging_steps == 0:
                    self._log_control_action(metrics)

                loss = total_loss_nats

            except Exception as e:
                self.logger.error(f"SCU control error at step {self.state.global_step}: {e}")
                loss = ce_loss_nats  # Fall back to standard loss
        else:
            loss = ce_loss_nats

        # Memory check and cleanup
        if self.state.global_step % self.memory_check_interval == 0:
            self._memory_cleanup_and_check()

        return (loss, outputs) if return_outputs else loss

    def _calculate_param_bpt_memory_efficient(self, model):
        """Calculate Parameter BPT with memory efficiency - maintains gradients"""
        # Start with zero on the correct device
        device = next(model.parameters()).device
        param_sum = torch.tensor(0.0, device=device, requires_grad=False)
        param_count = 0
        
        # Only process LoRA parameters to save memory
        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                # Sum of squares - move to float32 for numerical stability
                param_sum = param_sum + (param.float() ** 2).sum()
                param_count += param.numel()
        
        # Avoid division by zero
        if param_count == 0:
            return torch.tensor(1e-6, device=device, requires_grad=True)
        
        # Normalize (simplified calculation)
        tokens_per_epoch = 100000  # Fixed normalization
        param_bpt = param_sum / (2 * self.config.prior_sigma**2 * tokens_per_epoch * math.log(2))
        
        # Ensure requires_grad is True
        if not param_bpt.requires_grad:
            param_bpt.requires_grad_(True)
        
        return param_bpt

    def _log_control_action(self, metrics: Dict[str, Any]):
        """Log control action with memory info"""
        memory_mb = self._get_memory_usage_mb()
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        
        self.logger.info(
            f"SCU Step {metrics['step']}: "
            f"S={metrics['s_ratio']:.4f} ({metrics['s_ratio']*100:.2f}%), "
            f"λ={metrics['lambda']:.4f}, "
            f"Error={metrics['error']:.4f}, "
            f"Memory={memory_mb:.1f}MB"
        )

    def _memory_cleanup_and_check(self):
        """Perform memory cleanup and check for issues"""
        current_memory = self._get_memory_usage_mb()
        
        # Warning levels
        if current_memory > 35000:  # 35GB
            self.logger.warning(f"CRITICAL: Memory usage {current_memory:.1f} MB - approaching limit!")
            # Aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        elif current_memory > 30000:  # 30GB
            self.logger.warning(f"High memory usage: {current_memory:.1f} MB")
            gc.collect()

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def save_model(self, output_dir: str, _internal_call: bool = False):
        """Save model with SCU metadata and memory checks"""
        # Memory check before saving
        memory_before = self._get_memory_usage_mb()
        self.logger.info(f"Memory before save: {memory_before:.1f} MB")
        
        super().save_model(output_dir, _internal_call)
        
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
            'peak_memory_mb': self.peak_memory_mb
        }
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Memory check after saving
        memory_after = self._get_memory_usage_mb()
        self.logger.info(f"Memory after save: {memory_after:.1f} MB")
        
        # Cleanup after save
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def setup_logging(config: Granite1BSCUConfig) -> logging.Logger:
    """Setup memory-efficient logging configuration"""
    log_dir = Path(config.logging_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'memory_efficient_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MemoryEfficient-Granite-Training")


def load_and_prepare_dataset_memory_efficient(config: Granite1BSCUConfig, tokenizer):
    """Load and prepare dataset with memory efficiency focus"""
    try:
        # Load small dataset
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            split='train[:50000]'  # Limit to first 50K examples
        )
        
        # Create validation split
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({
            'train': dataset['train'],
            'validation': dataset['test']
        })
        
        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val")
        
        # Simple tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=config.block_size,
                return_overflowing_tokens=False,
            )
        
        # Tokenize with minimal workers
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=1,  # Single worker for memory efficiency
            remove_columns=dataset['train'].column_names,
            desc="Tokenizing dataset"
        )
        
        # Group texts into shorter blocks for memory efficiency
        def group_texts(examples):
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            # Use shorter blocks
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
        print(f"Dataset loading failed: {e}")
        raise


def create_model_and_tokenizer_memory_efficient(config: Granite1BSCUConfig, logger):
    """Create model and tokenizer with memory efficiency focus"""
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name,
            use_fast=config.use_fast_tokenizer,
            trust_remote_code=config.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
        
        # Load model with memory-efficient settings
        model_kwargs = {
            'torch_dtype': getattr(torch, config.torch_dtype),
            'trust_remote_code': config.trust_remote_code,
            'low_cpu_mem_usage': True,  # Critical for memory efficiency
        }
        
        # Device map for Apple Silicon or CPU
        if torch.cuda.is_available():
            model_kwargs['device_map'] = config.device_map
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            model_kwargs['device_map'] = 'mps'
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        logger.info(f"Base model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
        
        # Apply LoRA for memory efficiency
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,
                task_type=getattr(TaskType, config.lora_task_type)
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # Enable gradient checkpointing (critical for memory)
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Model creation failed: {e}")
        raise


def create_memory_efficient_config():
    """Create a memory-optimized configuration for testing"""
    config = Granite1BSCUConfig()
    
    # Ultra-conservative settings for memory testing
    config.batch_size = 1
    config.gradient_accumulation_steps = 32  # Higher accumulation
    config.block_size = 256  # Shorter sequences
    config.max_steps = 100  # Minimal test run
    config.lora_r = 4  # Minimal LoRA rank
    config.control_frequency = 10  # Less frequent control
    
    return config


def main():
    """Main memory-efficient training function"""
    parser = argparse.ArgumentParser(description="Memory-efficient SCU training for Granite-4.0-H-1B")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run with minimal steps for testing"
    )
    parser.add_argument(
        "--memory-safe",
        action="store_true",
        help="Use ultra-conservative memory settings"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./granite_1b_scu_output",
        help="Output directory"
    )
    parser.add_argument(
        "--config-override",
        type=str,
        help="JSON config override file"
    )

    args = parser.parse_args()

    # Load configuration
    config = Granite1BSCUConfig()
    config.output_dir = args.output_dir

    # Apply test run settings
    if args.test_run:
        config.max_steps = 50
        config.save_steps = 25
        config.eval_steps = 25
        config.logging_steps = 5
        config.warmup_steps = 5
        config.output_dir += "_test"

    # Apply memory-safe settings
    if args.memory_safe:
        config = create_memory_efficient_config()
        config.output_dir += "_safe"

    # Apply config override if provided
    if args.config_override:
        with open(args.config_override, 'r') as f:
            override = json.load(f)
        for key, value in override.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Setup logging
    global logger
    logger = setup_logging(config)
    logger.info("🧠 Starting memory-efficient SCU training for Granite-4.0-H-1B")
    logger.info(f"Target S ratio: {config.target_s_ratio:.3%}")
    logger.info(f"PI gains: Kp={config.kp}, Ki={config.ki}")
    logger.info(f"Max steps: {config.max_steps}")
    logger.info(f"Memory limit: 36GB (current system)")

    # Initial memory check
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    logger.info(f"Initial memory usage: {initial_memory:.1f} MB")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / "scu_config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    try:
        # Create model and tokenizer
        logger.info("Loading Granite-4.0-H-1B model and tokenizer...")
        model, tokenizer = create_model_and_tokenizer_memory_efficient(config, logger)
        logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
        
        # Memory check after model loading
        model_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory after model load: {model_memory:.1f} MB")

        # Load dataset
        logger.info("Loading and preparing dataset...")
        dataset = load_and_prepare_dataset_memory_efficient(config, tokenizer)
        logger.info(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['validation'])} val")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if config.fp16 else None
        )

        # Training arguments - memory optimized
        training_args = TrainingArguments(
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
            dataloader_num_workers=config.dataloader_num_workers,
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
            # Memory optimization
            gradient_checkpointing=config.gradient_checkpointing,
            # Disable some features for memory
            remove_unused_columns=False,
        )

        # Create memory-efficient trainer
        trainer = MemoryEfficientSCUTrainer(
            config=config,
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'] if config.run_validation else None,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if config.run_validation else None
        )

        # Initialize SCU
        trainer.initialize_scu()

        # Start training
        logger.info("🔬 Starting memory-efficient SCU training...")
        start_time = time.time()

        trainer.train()
        training_time = time.time() - start_time
        logger.info(f"SCU training completed in {training_time:.2f} seconds")

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {config.output_dir}")

        # Final summary
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info("🎯 Final Training Summary:")
        logger.info(f"  Total training time: {training_time:.2f} seconds")
        logger.info(f"  Peak memory usage: {trainer.peak_memory_mb:.1f} MB")
        logger.info(f"  Final memory usage: {final_memory:.1f} MB")
        logger.info(f"  Control actions applied: {len(trainer.control_metrics)}")
        
        if trainer.s_ratio_history:
            avg_s_ratio = sum(trainer.s_ratio_history) / len(trainer.s_ratio_history)
            logger.info(f"  Average S ratio: {avg_s_ratio:.4f} (target: {config.target_s_ratio:.4f})")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Final memory state on error
        error_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.error(f"Memory at error: {error_memory:.1f} MB")
        raise

    logger.info("🧠 Memory-efficient SCU training completed successfully!")


if __name__ == "__main__":
    main()