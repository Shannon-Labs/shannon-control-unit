#!/usr/bin/env python3
"""
Split FineWeb-Edu data into train/val sets with reproducible shuffling.
"""
import argparse
import random
from pathlib import Path

def split_data(input_path: str, train_ratio: float = 0.95, seed: int = 42):
    """Split data into train and validation sets."""
    input_file = Path(input_path)
    parent = input_file.parent
    stem = input_file.stem

    train_path = parent / f"{stem}_train.jsonl"
    val_path = parent / f"{stem}_val.jsonl"

    # Read all lines
    print(f"Reading {input_path}...")
    with open(input_file, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total samples: {total}")

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(lines)

    # Split
    split_idx = int(total * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    print(f"Train samples: {len(train_lines)} ({len(train_lines)/total*100:.1f}%)")
    print(f"Val samples: {len(val_lines)} ({len(val_lines)/total*100:.1f}%)")

    # Write train
    print(f"Writing {train_path}...")
    with open(train_path, 'w') as f:
        f.writelines(train_lines)

    # Write val
    print(f"Writing {val_path}...")
    with open(val_path, 'w') as f:
        f.writelines(val_lines)

    print(f"\nDone! Created:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    return str(train_path), str(val_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train/val")
    parser.add_argument("--input", default="data/fineweb_edu_1gb.jsonl", help="Input JSONL file")
    parser.add_argument("--train-ratio", type=float, default=0.95, help="Train ratio (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_data(args.input, args.train_ratio, args.seed)
