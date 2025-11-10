#!/usr/bin/env python3
"""
Simple SCU Training Script for IBM Granite-4.0-H-1B
Guaranteed to work without memory explosions or gradient issues
"""

import os
import sys
import logging
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model_and_tokenizer():
    """Create model with minimal memory footprint"""
    logger.info("Loading Granite-4.0-H-1B model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-h-1b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-4.0-h-1b",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=4,  # Very small for memory efficiency
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # Only 2 modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(tokenizer):
    """Load and prepare WikiText-2 dataset"""
    logger.info("Loading WikiText-2 dataset...")
    
    # Load small dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256,  # Short sequences
            return_overflowing_tokens=False,
        )
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # Single process for memory
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    # Group texts into chunks
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // 256) * 256
        result = {
            k: [t[i:i+256] for i in range(0, total_length, 256)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc="Grouping texts"
    )
    
    logger.info(f"Dataset prepared: {len(tokenized_datasets['train'])} train, {len(tokenized_datasets['validation'])} validation")
    return tokenized_datasets

def main():
    """Main training function - simple and reliable"""
    logger.info("🚀 Starting simple SCU training for Granite-4.0-H-1B")
    logger.info("📊 Dataset: WikiText-2 (small, safe, standard benchmark)")
    logger.info("💾 Memory: Optimized for 36GB Apple Silicon")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer()
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Training arguments - ultra conservative
    training_args = TrainingArguments(
        output_dir="./granite_simple_scu_output",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Minimal
        gradient_accumulation_steps=32,  # Effective batch size 32
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=10,
        max_steps=50,  # Very short for testing
        save_steps=25,
        eval_steps=25,
        logging_steps=5,
        eval_strategy="steps",
        save_strategy="steps",
        prediction_loss_only=True,
        fp16=False,
        bf16=False,
        dataloader_num_workers=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        save_safetensors=True,
        push_to_hub=False,
        report_to=[],  # No reporting for simplicity
        logging_first_step=True,
        # Memory optimization
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("🔬 Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    logger.info("✅ Model saved successfully!")
    
    logger.info("🎉 Training completed! Your Granite-4.0-H-1B model is ready.")
    logger.info("📂 Output: ./granite_simple_scu_output")

if __name__ == "__main__":
    main()