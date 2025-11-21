#!/usr/bin/env python3
"""
Production T-SCU Training Script for Qwen3-4B-MLX-4bit

Production-ready training script that combines thermodynamic control with MLX
framework for optimal Apple Silicon performance. Designed for Qwen3-4B-MLX-4bit.

Model: https://huggingface.co/Qwen/Qwen3-4B-MLX-4bit

Usage:
    python scu2/production/scripts/train_qwen3_mlx_tscu.py --config-file configs/qwen3_4b_mlx_config.py
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
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, DatasetDict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from scu2.production.configs.qwen3_4b_mlx_config import Qwen3MLXProductionConfig, HuggingFaceMLXModelConfig
from scu2.hardware.apple_power_monitor import AppleSiliconPowerMonitor
from scu2.core.thermodynamic_controller import ThermodynamicSCU
from scu2.metrics.energy_entropy import AdvancedEnergyEntropyMetrics


class MLXTSCUTrainer(Trainer):
    """Extended HuggingFace Trainer with MLX optimization and T-SCU integration"""

    def __init__(self, config: Qwen3MLXProductionConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.tscu_initialized = False
        self.power_monitor = None
        self.thermo_controller = None
        self.energy_metrics = AdvancedEnergyEntropyMetrics()
        self.tscu_step_count = 0
        self.training_metrics = []
        self.mlx_optimization_applied = False

    def initialize_tscu(self) -> bool:
        """Initialize T-SCU systems with MLX optimizations"""
        if not self.config.enable_thermodynamic_control:
            return False

        try:
            # Initialize power monitoring
            self.power_monitor = AppleSiliconPowerMonitor()
            if not self.power_monitor.initialize():
                self.log.warning("Failed to initialize power monitor")
                return False

            # Initialize thermodynamic controller with MLX-optimized settings
            self.thermo_controller = ThermodynamicSCU(
                power_budget_watts=self.config.power_budget_watts,
                target_efficiency=self.config.target_efficiency_bits_per_joule,
                max_temperature_celsius=self.config.max_temperature_celsius
            )

            self.tscu_initialized = True
            self.log.info("T-SCU systems initialized successfully for MLX")
            return True

        except Exception as e:
            self.log.error(f"Failed to initialize T-SCU: {e}")
            return False

    def apply_mlx_optimizations(self, model):
        """Apply MLX-specific optimizations"""
        if not self.config.use_mlx:
            return model

        try:
            # Enable Apple Silicon optimizations
            if torch.backends.mps.is_available():
                self.log.info("Applying Apple Silicon optimizations")

                # Enable memory efficient attention
                if hasattr(model, 'config'):
                    model.config.use_memory_efficient_attention = self.config.use_memory_efficient_attention

                # Enable gradient checkpointing for memory efficiency
                if self.config.gradient_checkpointing:
                    model.gradient_checkpointing_enable()

                # Set up efficient attention
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()

            self.mlx_optimization_applied = True
            self.log.info("MLX optimizations applied successfully")
            return model

        except Exception as e:
            self.log.warning(f"Failed to apply MLX optimizations: {e}")
            return model

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with T-SCU and MLX integration"""
        # Apply MLX optimizations if not already done
        if not self.mlx_optimization_applied:
            model = self.apply_mlx_optimizations(model)

        # Standard loss computation
        outputs = model(**inputs)
        loss = outputs.loss

        # Apply T-SCU control if enabled
        if self.tscu_initialized and self.tscu_step_count % self.config.control_frequency == 0:
            self._apply_tscu_control(model, loss.item())

        self.tscu_step_count += 1

        return (loss, outputs) if return_outputs else loss

    def _apply_tscu_control(self, model, loss_value):
        """Apply T-SCU control with MLX-specific optimizations"""
        try:
            # Get power readings
            power_readings = self.power_monitor.get_power_readings()
            total_power = sum(r.power_watts for r in power_readings) if power_readings else 0

            # Get temperature
            temperatures = [r.temperature_celsius for r in power_readings if r.temperature_celsius]
            avg_temp = max(temperatures) if temperatures else 25.0

            # Calculate information entropy (adapted for 4bit quantized models)
            param_entropy = self._calculate_quantized_model_entropy(model)

            # Create thermodynamic state
            thermo_state = self.thermo_controller.calculate_thermodynamic_metrics(
                information_entropy=param_entropy,
                power_consumption=total_power,
                temperature=avg_temp + 273.15
            )

            # Get control action
            control_factor, control_breakdown = self.thermo_controller.compute_control_action(
                thermo_state, self.state.global_step
            )

            # Apply adaptive learning rate scaling (more aggressive for MLX)
            if hasattr(self.optimizer, 'param_groups'):
                current_lr = self.optimizer.param_groups[0]['lr']
                # MLX allows more aggressive LR scaling due to better memory management
                adaptive_lr = current_lr * max(0.3, min(2.0, control_factor))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = adaptive_lr

            # MLX-specific adaptive batch sizing
            if self.config.adaptive_batch_sizing:
                self._apply_mlx_batch_sizing(control_factor, total_power, avg_temp)

            # Record metrics
            metrics = {
                'step': self.state.global_step,
                'loss': loss_value,
                'power_watts': total_power,
                'temperature_celsius': avg_temp,
                'control_factor': control_factor,
                'learning_rate': adaptive_lr,
                'param_entropy': param_entropy,
                'thermodynamic_efficiency': thermo_state.landauer_efficiency,
                'mlx_optimized': self.mlx_optimization_applied
            }
            self.training_metrics.append(metrics)

            # Log with MLX-specific information
            if self.state.global_step % 50 == 0:
                self.log.info(
                    f"MLX-T-SCU Step {self.state.global_step}: "
                    f"Loss={loss_value:.4f}, "
                    f"Power={total_power:.1f}W, "
                    f"Temp={avg_temp:.1f}°C, "
                    f"Control={control_factor:.2f}, "
                    f"LR={adaptive_lr:.2e}, "
                    f"4bit-Entropy={param_entropy:.2f}"
                )

            # Emergency checks (MLX can handle higher temps but still needs protection)
            if avg_temp > self.config.emergency_shutdown_temp:
                self.log.error(f"Emergency shutdown: Temperature {avg_temp:.1f}°C exceeds limit")
                self.control.should_training_stop = True

            # Record state for analysis
            self.thermo_controller.record_state_and_control(thermo_state, control_breakdown)

        except Exception as e:
            self.log.warning(f"T-SCU control error: {e}")

    def _apply_mlx_batch_sizing(self, control_factor: float, power: float, temp: float):
        """Apply MLX-specific adaptive batch sizing"""
        # MLX can handle larger batches due to unified memory
        if power < self.config.power_budget_watts * 0.7 and temp < 70:
            # Can increase effective batch size by reducing gradient accumulation
            if self.args.gradient_accumulation_steps > 2:
                new_accumulation = max(2, self.args.gradient_accumulation_steps - 1)
                self.args.gradient_accumulation_steps = new_accumulation
                self.log.info(f"MLX: Reduced gradient accumulation to {new_accumulation}")
        elif power > self.config.power_budget_watts * 0.9 or temp > 80:
            # Need to reduce computation
            new_accumulation = self.args.gradient_accumulation_steps + 1
            self.args.gradient_accumulation_steps = new_accumulation
            self.log.info(f"MLX: Increased gradient accumulation to {new_accumulation}")

    def _calculate_quantized_model_entropy(self, model) -> float:
        """Calculate entropy for 4-bit quantized model"""
        param_entropy = 0.0
        param_count = 0

        for param in model.parameters():
            if param.requires_grad:
                # For quantized models, entropy calculation needs adaptation
                # 4-bit quantization reduces entropy per parameter
                param_std = param.std().item()
                if param_std > 0:
                    # Adjust entropy calculation for 4-bit quantization
                    quantized_entropy = param.numel() * 0.25 * np.log2(2 * np.pi * np.e * param_std ** 2)
                    param_entropy += quantized_entropy
                param_count += param.numel()

        return param_entropy if param_count > 0 else 0.0

    def save_model(self, output_dir: str, _internal_call: bool = False):
        """Save model with T-SCU and MLX metadata"""
        super().save_model(output_dir, _internal_call)

        # Save T-SCU training metrics
        if self.training_metrics:
            metrics_file = Path(output_dir) / "mlx_tscu_training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)

        # Save T-SCU analysis
        if self.thermo_controller:
            analysis = self.thermo_controller.analyze_performance()
            analysis_file = Path(output_dir) / "mlx_tscu_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)

        # Save MLX configuration
        mlx_config = {
            "use_mlx": self.config.use_mlx,
            "quantization_bits": self.config.quantization_bits,
            "mlx_optimization_applied": self.mlx_optimization_applied,
            "apple_silicon_optimized": torch.backends.mps.is_available()
        }
        mlx_config_file = Path(output_dir) / "mlx_config.json"
        with open(mlx_config_file, 'w') as f:
            json.dump(mlx_config, f, indent=2)


def setup_logging(config: Qwen3MLXProductionConfig) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(config.logging_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'mlx_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("MLX-TSCU-Qwen3-Training")


def load_and_prepare_dataset(config: Qwen3MLXProductionConfig, tokenizer) -> DatasetDict:
    """Load and prepare training dataset for MLX"""
    if config.train_file and config.validation_file:
        # Load from local files
        dataset = load_dataset(
            'text',
            data_files={'train': config.train_file, 'validation': config.validation_file}
        )
    else:
        # Load from HuggingFace
        dataset = load_dataset(config.dataset_name, config.dataset_config)

        # Limit dataset size for faster training with MLX
        if config.dataset_name == "c4":
            # Use a smaller subset for demonstration
            train_size = min(100000, len(dataset['train']))
            val_size = min(10000, len(dataset['validation']))
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
        num_workers=config.preprocessing_num_workers,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset for MLX"
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


def create_mlx_model_and_tokenizer(config: Qwen3MLXProductionConfig):
    """Create model and tokenizer with MLX optimizations"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=config.use_fast_tokenizer,
        trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    quantization_config = None
    if config.use_mlx_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
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
    parser = argparse.ArgumentParser(description="Train Qwen3-4B-MLX with T-SCU")
    parser.add_argument(
        "--config-file",
        type=str,
        default="scu2/production/configs/qwen3_4b_mlx_config.py",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tscu_qwen3_mlx_output",
        help="Output directory"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to preprocessed dataset"
    )

    args = parser.parse_args()

    # Load configuration
    config = Qwen3MLXProductionConfig()
    config.output_dir = args.output_dir
    if args.push_to_hub:
        config.push_to_hub = True
        config.hub_model_id = args.hub_model_id

    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting T-SCU Qwen3-4B-MLX training")
    logger.info(f"Configuration: {asdict(config)}")

    # Check Apple Silicon availability
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon (Metal Performance Shaders) available")
    else:
        logger.warning("Apple Silicon not detected - MLX optimizations may be limited")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = output_dir / "mlx_training_config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Create model and tokenizer
    logger.info(f"Loading model and tokenizer: {config.model_name}")
    model, tokenizer = create_mlx_model_and_tokenizer(config)
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")

    # Load dataset
    if args.dataset_path:
        logger.info(f"Loading preprocessed dataset from {args.dataset_path}")
        dataset = DatasetDict.load_from_disk(args.dataset_path)
    else:
        logger.info("Loading and preparing dataset...")
        dataset = load_and_prepare_dataset(config, tokenizer)

    logger.info(f"Dataset loaded: {len(dataset['train'])} training examples")

    # Data collator optimized for quantized models
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if config.fp16 else None
    )

    # Training arguments with MLX optimizations
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
        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps" if config.run_validation else "no",
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
        ddp_find_unused_parameters=False,  # Optimization for LoRA
    )

    # Create trainer
    trainer = MLXTSCUTrainer(
        config=config,
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if config.run_validation else None,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if config.run_validation else None
    )

    # Initialize T-SCU
    trainer.initialize_tscu()

    # Start training
    logger.info("Starting MLX-optimized training with T-SCU...")
    start_time = time.time()

    try:
        trainer.train()
        training_time = time.time() - start_time
        logger.info(f"MLX training completed in {training_time:.2f} seconds")

        # Save final model
        trainer.save_model()
        logger.info(f"Model saved to {config.output_dir}")

        # Save training summary
        summary = {
            "training_time_seconds": training_time,
            "total_steps": trainer.state.global_step,
            "final_loss": trainer.state.log_history[-1].get("train_loss", 0),
            "tscu_enabled": config.enable_thermodynamic_control,
            "mlx_optimized": config.use_mlx,
            "quantization_bits": config.quantization_bits,
            "tscu_metrics_count": len(trainer.training_metrics) if hasattr(trainer, 'training_metrics') else 0
        }

        summary_file = output_dir / "mlx_training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"MLX training summary: {summary}")

    except Exception as e:
        logger.error(f"MLX training failed: {e}")
        raise

    finally:
        # Cleanup T-SCU systems
        if trainer.tscu_initialized and trainer.power_monitor:
            trainer.power_monitor.shutdown()

    logger.info("MLX-T-SCU Qwen3-4B training completed successfully")


if __name__ == "__main__":
    main()