#!/usr/bin/env python3
"""
Simplified Qwen3-4B-MLX-4bit Training Script

Clean, focused training script that uses the simplified SCU for computational
efficiency optimization. No unnecessary thermodynamic simulation.

Model: https://huggingface.co/Qwen/Qwen3-4B-MLX-4bit
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from dataclasses import asdict

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
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from scu2.core.simplified_controller import SimplifiedSCU, TrainingState
from scu2.production.configs.qwen3_4b_mlx_config import Qwen3MLXProductionConfig


class SimplifiedTrainer(Trainer):
    """Extended HuggingFace Trainer with simplified SCU integration"""

    def __init__(self, config: Qwen3MLXProductionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.sscu = SimplifiedSCU(
            target_loss_improvement=0.01,
            control_frequency=50,  # Control every 50 steps
            enable_multiscale_analysis=True
        )
        self.training_metrics = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with simplified SCU integration"""
        # Ensure inputs are on the correct device and require gradients
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(model.device)

        # Standard loss computation
        outputs = model(**inputs)
        loss = outputs.loss

        # Ensure loss requires gradients
        if not loss.requires_grad:
            loss = loss.mean()
            if hasattr(loss, 'requires_grad'):
                loss.requires_grad_(True)

        # Get gradient norm for SCU analysis (only after backward pass)
        total_norm = 0.0
        if hasattr(model, 'parameters') and self.optimizer:
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

        # Create training state
        current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
        training_state = TrainingState(
            loss_value=loss.item(),
            gradient_norm=total_norm,
            learning_rate=current_lr,
            batch_size=self.args.train_batch_size,
            step_count=self.state.global_step,
            timestamp=time.time()
        )

        # Get SCU recommendation
        action = self.sscu.update_state(training_state)

        # Apply control action if recommended
        if action and self.optimizer:
            self._apply_control_action(action)

            # Log the action
            if self.state.global_step % 100 == 0:
                self.log.info(f"SCU Step {self.state.global_step}: {action.reason}")

        return (loss, outputs) if return_outputs else loss

    def _apply_control_action(self, action):
        """Apply SCU control action to training"""
        if hasattr(self.optimizer, 'param_groups'):
            # Adjust learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = current_lr * action.learning_rate_factor
            new_lr = max(1e-7, min(1e-3, new_lr))  # Reasonable bounds
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            # Adjust gradient clipping if needed
            if hasattr(self.args, 'max_grad_norm') and action.gradient_clipping_factor != 1.0:
                self.args.max_grad_norm *= action.gradient_clipping_factor

        # Store action in metrics
        self.training_metrics.append({
            'step': self.state.global_step,
            'action': action.reason,
            'lr_factor': action.learning_rate_factor,
            'batch_adjustment': action.batch_size_adjustment,
            'grad_clip_factor': action.gradient_clipping_factor,
            'reg_factor': action.regularization_factor
        })

    def save_model(self, output_dir: str, _internal_call: bool = False):
        """Save model with SCU training metrics"""
        super().save_model(output_dir, _internal_call)

        # Save SCU training metrics and summary
        if self.training_metrics:
            metrics_file = Path(output_dir) / "scu_training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)

        # Save SCU summary
        summary = self.sscu.get_training_summary()
        summary_file = Path(output_dir) / "scu_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def setup_logging(config: Qwen3MLXProductionConfig) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(config.logging_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'simple_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Simple-SCU-Qwen3-Training")


def load_and_prepare_dataset(config: Qwen3MLXProductionConfig, tokenizer) -> DatasetDict:
    """Load and prepare training dataset"""
    if config.train_file and config.validation_file:
        # Load from local files
        dataset = load_dataset(
            'text',
            data_files={'train': config.train_file, 'validation': config.validation_file}
        )
    else:
        # Load from HuggingFace
        dataset = load_dataset(config.dataset_name, config.dataset_config)

        # Use a reasonable subset for demonstration
        if config.dataset_name == "c4":
            train_size = min(50000, len(dataset['train']))
            val_size = min(5000, len(dataset['validation']))
            dataset = DatasetDict({
                'train': dataset['train'].select(range(train_size)),
                'validation': dataset['validation'].select(range(val_size))
            })

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=config.block_size,
            return_overflowing_tokens=False,
        )

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
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


def create_model_and_tokenizer(config: Qwen3MLXProductionConfig):
    """Create model and tokenizer"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=config.use_fast_tokenizer,
        trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (without quantization on macOS)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True
    )

    # Apply LoRA if specified
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

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Qwen3-4B-MLX with Simplified SCU")
    parser.add_argument(
        "--config-file",
        type=str,
        default="scu2/production/configs/qwen3_4b_mlx_config.py",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./simple_scu_qwen3_output",
        help="Output directory"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to preprocessed dataset"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run with minimal dataset for testing"
    )

    args = parser.parse_args()

    # Load configuration
    config = Qwen3MLXProductionConfig()
    config.output_dir = args.output_dir

    if args.test_run:
        config.max_steps = 100
        config.save_steps = 50
        config.eval_steps = 50
        config.logging_steps = 10

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Simplified SCU Qwen3-4B-MLX training")
    logger.info(f"Configuration: {asdict(config)}")

    # Check Apple Silicon availability
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon (Metal Performance Shaders) available")
    else:
        logger.warning("Apple Silicon not detected - training may be slower")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / "simple_training_config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Create model and tokenizer
    logger.info(f"Loading model and tokenizer: {config.model_name}")
    model, tokenizer = create_model_and_tokenizer(config)
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")

    # Load dataset
    if args.dataset_path:
        logger.info(f"Loading preprocessed dataset from {args.dataset_path}")
        dataset = DatasetDict.load_from_disk(args.dataset_path)
    else:
        logger.info("Loading and preparing dataset...")
        dataset = load_and_prepare_dataset(config, tokenizer)

    logger.info(f"Dataset loaded: {len(dataset['train'])} training examples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if config.fp16 else None
    )

    # Training arguments
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
        eval_steps=config.eval_steps,
        eval_strategy="steps" if config.run_validation else "no",
        prediction_loss_only=True,
        fp16=config.fp16 and not torch.backends.mps.is_available(),
        bf16=config.bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        save_total_limit=config.save_total_limit,
        save_safetensors=config.save_safetensors,
        report_to=[],  # Disable TensorBoard for test run
        logging_first_step=config.logging_first_step,
        ddp_find_unused_parameters=False,
    )

    # Create trainer
    trainer = SimplifiedTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if config.run_validation else None,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if config.run_validation else None
    )

    # Start training
    logger.info("Starting training with Simplified SCU...")
    start_time = time.time()

    try:
        trainer.train()
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {config.output_dir}")

        # Save training summary
        summary = {
            "training_time_seconds": training_time,
            "total_steps": trainer.state.global_step,
            "final_loss": trainer.state.log_history[-1].get("train_loss", 0),
            "scu_enabled": True,
            "scu_actions_applied": len(trainer.training_metrics) if hasattr(trainer, 'training_metrics') else 0,
            "test_run": args.test_run
        }

        summary_file = output_dir / "simple_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary: {summary}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info("Simplified SCU Qwen3-4B training completed successfully")


if __name__ == "__main__":
    main()