#!/usr/bin/env python3
"""
Ultra-Active SCU Training Script for IBM Granite-4.0-Micro

This script implements the most aggressive information entropy control for maximum
training efficiency. The SCU makes micro-adjustments EVERY training step.

Usage:
    python scu2/production/scripts/train_granite_ultra_scu.py [--test-run]
"""

import os
import sys
import argparse
import logging
import json
import time
import math
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

from scu2.production.configs.granite_micro_ultra_config import GraniteMicroUltraSCUConfig, GraniteModelCardConfig
from scu2.core.ultra_active_controller import UltraActiveSCU, UltraTrainingState


class UltraActiveSCUTrainer(Trainer):
    """Trainer with ultra-active SCU integration for maximum efficiency"""

    def __init__(self, config: GraniteMicroUltraSCUConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.scu_initialized = False
        self.ultra_scu = None
        self.prior_sigma = config.prior_sigma
        self.tokens_per_epoch = 1000000  # Fixed normalization constant

        # Training metrics
        self.training_metrics = []
        self.s_ratio_history = []
        self.lambda_history = []
        self.control_frequency_stats = []

  
        # Setup logger
        import logging
        self.logger = logging.getLogger("UltraActive-Trainer")

    def initialize_ultra_scu(self) -> bool:
        """Initialize ultra-active SCU system"""
        try:
            self.ultra_scu = UltraActiveSCU(
                target_s_ratio=self.config.target_s_ratio,
                lambda_init=self.config.lambda_init,
                lambda_min=self.config.lambda_min,
                lambda_max=self.config.lambda_max,
                kp=self.config.kp,
                ki=self.config.ki,
                deadband=self.config.deadband,
                ema_alpha=self.config.ema_alpha,
                integral_leak=self.config.integral_leak,
                enable_adaptive_gains=self.config.enable_adaptive_gains,
                enable_predictive_control=self.config.enable_predictive_control,
                adaptive_kp_range=self.config.adaptive_kp_range,
                adaptive_ki_range=self.config.adaptive_ki_range,
                prediction_horizon=self.config.prediction_horizon,
              )

            self.scu_initialized = True
            self.logger.info("Ultra-Active SCU initialized successfully")
            self.logger.info(f"Target S ratio: {self.config.target_s_ratio:.3%}")
            self.logger.info(f"PI gains: Kp={self.config.kp}, Ki={self.config.ki}")
            self.logger.info(f"Control frequency: EVERY STEP (ultra-active)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Ultra-Active SCU: {e}")
            return False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with ultra-active SCU integration"""

        # Standard forward pass
        outputs = model(**inputs)
        ce_loss_nats = outputs.loss

        # Apply SCU control if enabled (EVERY STEP)
        if self.scu_initialized and self.ultra_scu:
            # Calculate data BPT
            data_bpt = ce_loss_nats / math.log(2)

            # Calculate parameter BPT for LoRA weights
            param_bpt = self._calculate_param_bpt(model)

            # Calculate S ratio
            total_bpt = data_bpt + param_bpt
            s_ratio = (param_bpt / total_bpt).item() if total_bpt > 0 else 0.0

            # Get gradient norm
            grad_norm = self._get_gradient_norm(model)

            # Create ultra training state
            state = UltraTrainingState(
                loss_value=ce_loss_nats.item(),
                data_bpt=data_bpt,
                param_bpt=param_bpt,
                s_ratio=s_ratio,
                gradient_norm=grad_norm,
                learning_rate=self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
                lambda_value=self.ultra_scu.lambda_current,
                step_count=self.state.global_step,
                timestamp=time.time()
            )

            # Get ultra-active control action
            control_action = self.ultra_scu.update_state(state)

            # Apply lambda regularization - param_bpt already has gradients
            reg_loss_nats = param_bpt * math.log(2) * self.tokens_per_epoch  # Still has gradients
            # Apply lambda - both terms have gradients now
            total_loss_nats = ce_loss_nats + control_action.new_lambda * reg_loss_nats

            # Store metrics
            metrics = {
                'step': self.state.global_step,
                'ce_loss': ce_loss_nats.item(),
                'reg_loss': reg_loss_nats.item(),
                'total_loss': total_loss_nats.item(),
                'data_bpt': data_bpt,
                'param_bpt': param_bpt.item(),
                's_ratio': s_ratio,
                'lambda': control_action.new_lambda,
                'control_effort': control_action.control_effort,
                'adaptive_kp': control_action.adaptive_kp,
                'adaptive_ki': control_action.adaptive_ki
            }

            self.training_metrics.append(metrics)
            self.s_ratio_history.append(s_ratio)
            self.lambda_history.append(control_action.new_lambda)

            # Log frequently for ultra-active monitoring
            if self.state.global_step % self.config.logging_steps == 0:
                self._log_ultra_active_metrics(metrics, control_action)

            loss = total_loss_nats

        else:
            loss = ce_loss_nats

        return (loss, outputs) if return_outputs else loss

    def _calculate_param_bpt(self, model):
        """Calculate Parameter BPT for LoRA weights - returns tensor with gradients"""
        param_sum = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                # Sum of squares maintaining gradients
                param_sum = param_sum + (param.float() ** 2).sum()

        # Convert to bits and normalize - maintain gradients
        param_bpt = param_sum / (2 * self.prior_sigma**2 * self.tokens_per_epoch * math.log(2))
        return param_bpt

    def _get_gradient_norm(self, model) -> float:
        """Get current gradient norm"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

  
    def _log_ultra_active_metrics(self, metrics: dict, control_action):
        """Log detailed ultra-active metrics"""
        self.logger.info(
            f"ULTRA-ACTIVE SCU Step {self.state.global_step}: "
            f"Loss={metrics['total_loss']:.4f}, "
            f"S={metrics['s_ratio']:.4f} ({metrics['s_ratio']*100:.2f}%), "
            f"λ={metrics['lambda']:.4f}, "
            f"Effort={control_action.control_effort:.4f}, "
            f"Kp={control_action.adaptive_kp:.2f}, "
            f"Ki={control_action.adaptive_ki:.2f}"
        )

        if self.config.enable_detailed_logging:
            self.logger.info(f"  Control reason: {control_action.reason}")
            if control_action.detailed_breakdown:
                details = control_action.detailed_breakdown
                self.logger.info(
                    f"  Details: Error={details.get('error', 0):.4f}, "
                    f"Integral={details.get('integral_term', 0):.4f}, "
                    f"Prediction={details.get('prediction_adjustment', 0):.4f}"
                )

    def save_model(self, output_dir: str, _internal_call: bool = False):
        """Save model with ultra-active SCU metadata"""
        super().save_model(output_dir, _internal_call)

        # Save ultra-active SCU training metrics
        if self.training_metrics:
            metrics_file = Path(output_dir) / "ultra_active_scu_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)

        # Save S ratio and lambda history
        history_file = Path(output_dir) / "scu_history.json"
        history_data = {
            's_ratio_history': self.s_ratio_history,
            'lambda_history': self.lambda_history,
            'target_s_ratio': self.config.target_s_ratio
        }
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)

        # Save ultra-active SCU analysis
        if self.ultra_scu:
            summary = self.ultra_scu.get_comprehensive_summary()
            summary_file = Path(output_dir) / "ultra_active_scu_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Ultra-Active SCU Summary: {summary}")


def setup_logging(config: GraniteMicroUltraSCUConfig) -> logging.Logger:
    """Setup enhanced logging configuration"""
    log_dir = Path(config.logging_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'ultra_active_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("UltraActive-Granite-Training")


# Module-level functions for streaming datasets (to fix pickling issues)
def tokenize_and_chunk_streaming(example, tokenizer, block_size):
    """Tokenize and chunk streaming examples - moved to module level for pickling"""
    # Handle different example formats
    if isinstance(example, dict):
        if 'text' in example:
            text = example['text']
        elif len(example) == 1:
            text = list(example.values())[0]
        else:
            # For wikitext, the text might be under different key
            # Get the first string value
            text = None
            for value in example.values():
                if isinstance(value, str):
                    text = value
                    break
            if text is None:
                raise ValueError(f"No string found in example: {example}")
    elif isinstance(example, str):
        text = example
    elif isinstance(example, list) and len(example) > 0:
        # Handle case where example is a list
        text = str(example[0]) if isinstance(example[0], str) else str(example)
    else:
        text = str(example)

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=block_size,
        return_overflowing_tokens=False,
    )

    # Simple chunking for streaming
    input_ids = tokenized['input_ids']

    # If we have enough tokens for a full chunk, return it
    if len(input_ids) >= block_size:
        chunk = input_ids[:block_size]
        return {
            'input_ids': chunk,
            'labels': chunk.copy(),
            'attention_mask': [1] * len(chunk)
        }

    # If tokens are shorter than block_size, pad them
    if len(input_ids) > 0:
        pad_length = block_size - len(input_ids)
        chunk = input_ids + [tokenizer.pad_token_id or 0] * pad_length
        return {
            'input_ids': chunk,
            'labels': chunk.copy(),
            'attention_mask': [1] * len(input_ids) + [0] * pad_length
        }

    # Return empty chunk if no tokens
    return {
        'input_ids': [tokenizer.pad_token_id or 0] * block_size,
        'labels': [tokenizer.pad_token_id or 0] * block_size,
        'attention_mask': [0] * block_size
    }


def load_and_prepare_dataset(config: GraniteMicroUltraSCUConfig, tokenizer) -> DatasetDict:
    """Load and prepare training dataset with STREAMING for efficiency"""
    if config.train_file and config.validation_file:
        # Load from local files
        dataset = load_dataset(
            'text',
            data_files={'train': config.train_file, 'validation': config.validation_file}
        )
    else:
        # STREAM from HuggingFace - no massive downloads!
        print(f"🚀 Streaming from {config.dataset_name}/{config.dataset_config}")
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
            streaming=True,  # KEY: Stream, don't download everything!
            split='train'
        )

        # Take only what we need
        if hasattr(config, 'max_examples') and config.max_examples:
            print(f"📊 Taking {config.max_examples} examples from stream")
            dataset = dataset.take(config.max_examples)
        else:
            # Default: 10K for test, 100K for full training
            max_examples = 1000 if config.max_steps <= 100 else 100000
            print(f"📊 Taking {max_examples} examples from stream")
            dataset = dataset.take(max_examples)

        # Convert back to DatasetDict for compatibility
        dataset = DatasetDict({'train': dataset})

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=config.block_size,
            return_overflowing_tokens=False,
        )

    # Check if this is a streaming dataset (IterableDataset)
    from datasets import IterableDataset
    is_streaming = isinstance(dataset.get('train') if hasattr(dataset, 'get') else dataset, IterableDataset)

    if not is_streaming:
        # Regular dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=config.preprocessing_num_workers,
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
    else:
        # Streaming dataset - handle differently
        print("Processing streaming dataset...")

        # Get the actual iterable dataset
        stream_dataset = dataset.get('train') if hasattr(dataset, 'get') else dataset

        # Apply tokenization and chunking to streaming dataset using module-level function
        processed_dataset = stream_dataset.map(
            lambda example: tokenize_and_chunk_streaming(example, tokenizer, config.block_size),
            remove_columns=['text']
        )

        # For streaming, we need to work directly with the iterable
        grouped_dataset = processed_dataset

    return grouped_dataset


def create_model_and_tokenizer(config: GraniteMicroUltraSCUConfig):
    """Create Granite model and tokenizer"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=config.use_fast_tokenizer,
        trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code
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
    """Main ultra-active training function"""
    parser = argparse.ArgumentParser(description="Ultra-Active SCU training for Granite-4.0-Micro")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run with minimal steps for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./granite_micro_ultra_scu_output",
        help="Output directory"
    )
    parser.add_argument(
        "--config-override",
        type=str,
        help="JSON config override"
    )

    args = parser.parse_args()

    # Load configuration
    config = GraniteMicroUltraSCUConfig()
    config.output_dir = args.output_dir

    # Apply test run settings
    if args.test_run:
        config.max_steps = 50
        config.save_steps = 25
        config.eval_steps = 25
        config.logging_steps = 5
        config.warmup_steps = 5
        config.output_dir += "_test"

    # Apply config override if provided
    if args.config_override:
        with open(args.config_override, 'r') as f:
            override = json.load(f)
        for key, value in override.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Setup logging
    logger = setup_logging(config)
    logger.info("🚀 Starting Ultra-Active SCU training for Granite-4.0-Micro")
    logger.info(f"Ultra-Active Mode: Control EVERY step")
    logger.info(f"Target S ratio: {config.target_s_ratio:.3%}")
    logger.info(f"PI gains: Kp={config.kp}, Ki={config.ki}")
    logger.info(f"Max steps: {config.max_steps}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / "ultra_active_scu_config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Create model and tokenizer
    logger.info("Loading Granite-4.0-Micro model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(config)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")

    # Load dataset
    logger.info("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(config, tokenizer)

    # Check if this is a streaming dataset
    from datasets import IterableDataset
    is_streaming = isinstance(dataset.get('train') if hasattr(dataset, 'get') else dataset, IterableDataset)

    # Handle different dataset types for logging
    try:
        train_examples = len(dataset['train'])
        logger.info(f"Dataset loaded: {train_examples} training examples")
    except (NotImplementedError, TypeError):
        # Streaming dataset - estimate size
        logger.info("Dataset loaded: streaming dataset")

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
    )

    # Handle dataset assignment for streaming
    train_dataset = dataset
    eval_dataset = None

    # For streaming datasets, we need to extract the actual iterable
    if is_streaming:
        # Already processed the streaming dataset
        pass
    else:
        # Regular dataset
        train_dataset = dataset['train']
        eval_dataset = dataset['validation'] if config.run_validation else None

    # Create ultra-active trainer
    trainer = UltraActiveSCUTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if config.run_validation else None
    )

    # Initialize ultra-active SCU
    trainer.initialize_ultra_scu()

    # Start training
    logger.info("🔥 Starting ultra-active training with maximum efficiency optimization...")
    start_time = time.time()

    try:
        trainer.train()
        training_time = time.time() - start_time
        logger.info(f"Ultra-Active training completed in {training_time:.2f} seconds")

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {config.output_dir}")

        # Save training summary
        if trainer.ultra_scu:
            summary = trainer.ultra_scu.get_comprehensive_summary()
            summary_file = output_dir / "final_ultra_active_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            # Calculate and log efficiency metrics
            control_frequency = summary['control_frequency_percent']
            final_s_ratio = summary['current_s_ratio']
            target_s_ratio = summary['target_s_ratio']
            s_error = abs(final_s_ratio - target_s_ratio) / target_s_ratio * 100

            logger.info("🎯 Ultra-Active SCU Performance Summary:")
            logger.info(f"  Control frequency: {control_frequency:.1f}% of steps")
            logger.info(f"  Final S ratio: {final_s_ratio:.4f} (target: {target_s_ratio:.4f})")
            logger.info(f"  S ratio error: {s_error:.2f}%")
            logger.info(f"  Total control actions: {summary['control_actions_applied']}")
            logger.info(f"  Efficiency estimate: {summary['efficiency_estimate']:.2f}x")
            logger.info(f"  Lambda volatility: {summary['lambda_volatility']:.4f}")

    except Exception as e:
        logger.error(f"Ultra-Active training failed: {e}")
        raise

    logger.info("🚀 Ultra-Active SCU training completed successfully!")


if __name__ == "__main__":
    main()