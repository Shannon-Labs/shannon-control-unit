#!/usr/bin/env python3
"""
Dataset Preparation for T-SCU Qwen3:4B Training

Prepares training datasets with proper tokenization and formatting for Qwen3-4B
training with thermodynamic control.

Usage:
    python scu2/production/scripts/prepare_dataset.py --dataset-name wikitext --output-dir ./data
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("DatasetPreparation")


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def prepare_wikitext_dataset(
    tokenizer: AutoTokenizer,
    block_size: int = 1024,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """Prepare WikiText-103 dataset"""

    # Load dataset
    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        cache_dir=cache_dir
    )

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=block_size,
            return_overflowing_tokens=False,
        )

    # Apply tokenization
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_workers=4,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset"
    )

    # Group texts into blocks
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {block_size}"
    )

    return grouped_datasets


def prepare_c4_dataset(
    tokenizer: AutoTokenizer,
    block_size: int = 1024,
    cache_dir: Optional[str] = None,
    dataset_size: Optional[int] = None
) -> DatasetDict:
    """Prepare C4 dataset (subset for faster training)"""

    # Load C4 dataset
    dataset = load_dataset(
        "c4",
        "en",
        cache_dir=cache_dir
    )

    # Limit dataset size for faster training
    if dataset_size:
        dataset = DatasetDict({
            'train': dataset['train'].select(range(min(dataset_size, len(dataset['train'])))),
            'validation': dataset['validation'].select(range(min(dataset_size // 10, len(dataset['validation']))))
        })

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=block_size,
            return_overflowing_tokens=False,
        )

    # Apply tokenization
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_workers=4,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset"
    )

    # Group texts into blocks
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {block_size}"
    )

    return grouped_datasets


def prepare_custom_dataset(
    tokenizer: AutoTokenizer,
    data_file: str,
    block_size: int = 1024,
    validation_split: float = 0.1
) -> DatasetDict:
    """Prepare custom dataset from file"""

    # Load text file
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into chunks (simple approach)
    chunk_size = block_size * 10  # Larger chunks for better context
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Split into train/validation
    split_idx = int(len(text_chunks) * (1 - validation_split))
    train_chunks = text_chunks[:split_idx]
    val_chunks = text_chunks[split_idx:]

    # Create dataset
    from datasets import Dataset
    train_dataset = Dataset.from_dict({'text': train_chunks})
    val_dataset = Dataset.from_dict({'text': val_chunks})

    dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=block_size,
            return_overflowing_tokens=False,
        )

    # Apply tokenization
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_workers=4,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing dataset"
    )

    # Group texts into blocks
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {block_size}"
    )

    return grouped_datasets


def main():
    """Main dataset preparation function"""
    parser = argparse.ArgumentParser(description="Prepare dataset for T-SCU Qwen3-4B training")
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["wikitext", "c4", "custom"],
        default="wikitext",
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to custom data file (required for custom dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./scu2/production/data",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Block size for tokenization"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory for datasets"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        help="Limit dataset size (for testing)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info(f"Preparing {args.dataset_name} dataset for T-SCU Qwen3-4B training")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)

    # Prepare dataset
    if args.dataset_name == "wikitext":
        logger.info("Preparing WikiText-103 dataset")
        dataset = prepare_wikitext_dataset(
            tokenizer=tokenizer,
            block_size=args.block_size,
            cache_dir=args.cache_dir
        )
    elif args.dataset_name == "c4":
        logger.info("Preparing C4 dataset")
        dataset = prepare_c4_dataset(
            tokenizer=tokenizer,
            block_size=args.block_size,
            cache_dir=args.cache_dir,
            dataset_size=args.dataset_size
        )
    elif args.dataset_name == "custom":
        if not args.data_file:
            raise ValueError("--data-file required for custom dataset")
        logger.info(f"Preparing custom dataset from {args.data_file}")
        dataset = prepare_custom_dataset(
            tokenizer=tokenizer,
            data_file=args.data_file,
            block_size=args.block_size
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Save dataset
    dataset_path = output_dir / f"{args.dataset_name}_prepared"
    logger.info(f"Saving prepared dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)

    # Log statistics
    train_size = len(dataset['train'])
    val_size = len(dataset['validation'])
    total_tokens = train_size * args.block_size + val_size * args.block_size

    logger.info(f"Dataset prepared successfully:")
    logger.info(f"  Training examples: {train_size}")
    logger.info(f"  Validation examples: {val_size}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Block size: {args.block_size}")
    logger.info(f"  Saved to: {dataset_path}")

    # Save dataset info
    dataset_info = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "block_size": args.block_size,
        "train_size": train_size,
        "validation_size": val_size,
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.vocab_size
    }

    info_file = dataset_path / "dataset_info.json"
    import json
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info("Dataset preparation completed successfully")


if __name__ == "__main__":
    main()