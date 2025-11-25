#!/usr/bin/env python3
"""
FineWeb-Edu Dataset Loader for SCU Training

Downloads a subset of allenai/fineweb-edu and prepares it for training.
The dataset is high-quality educational web content, ideal for evaluating
MDL-based training approaches.

Usage:
    python scripts/load_fineweb_edu.py --size 50  # Download ~50MB
    python scripts/load_fineweb_edu.py --size 500 --output data/fineweb_edu_500mb.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import sys


def download_fineweb_edu(
    target_size_mb: float = 50.0,
    output_path: str = "data/fineweb_edu_sample.jsonl",
    split: str = "train",
    seed: int = 42,
    min_score: float = 3.0,
    max_texts: Optional[int] = None,
    streaming: bool = True,
) -> dict:
    """
    Download and save a subset of FineWeb-Edu.

    Args:
        target_size_mb: Target size in MB (approximate)
        output_path: Output JSONL file path
        split: Dataset split (train/val)
        seed: Random seed for shuffling
        min_score: Minimum educational score (0-5, default 3.0)
        max_texts: Maximum number of texts to download
        streaming: Use streaming mode (recommended for large datasets)

    Returns:
        Dictionary with statistics about downloaded data
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed.")
        print("Run: pip install datasets")
        sys.exit(1)

    print(f"Loading FineWeb-Edu (target: {target_size_mb}MB)...")

    # FineWeb-Edu is large, so we use streaming mode
    # The dataset has 'text' field and 'score' field
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # Use the 10BT sample for manageable size
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "total_texts": 0,
        "total_chars": 0,
        "total_bytes": 0,
        "min_score_filter": min_score,
        "texts_filtered": 0,
        "avg_text_length": 0,
    }

    target_bytes = int(target_size_mb * 1024 * 1024)

    print(f"Downloading to: {output_path}")
    print(f"Filtering for score >= {min_score}")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            # Check if we've reached target size or max texts
            if stats["total_bytes"] >= target_bytes:
                print(f"\nReached target size: {stats['total_bytes'] / 1024 / 1024:.1f}MB")
                break
            if max_texts and stats["total_texts"] >= max_texts:
                print(f"\nReached max texts: {max_texts}")
                break

            # Filter by educational score if available
            score = example.get("score", 5.0)  # Default high score if not present
            if score < min_score:
                stats["texts_filtered"] += 1
                continue

            # Extract text
            text = example.get("text", "")
            if not text or len(text) < 100:  # Skip very short texts
                continue

            # Write as JSONL
            record = {
                "text": text,
                "score": score,
                "id": example.get("id", f"fineweb_{i}"),
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            f.write(line)

            stats["total_texts"] += 1
            stats["total_chars"] += len(text)
            stats["total_bytes"] += len(line.encode("utf-8"))

            # Progress
            if stats["total_texts"] % 100 == 0:
                mb = stats["total_bytes"] / 1024 / 1024
                print(f"\rDownloaded: {stats['total_texts']} texts, {mb:.1f}MB", end="")

    # Calculate final statistics
    if stats["total_texts"] > 0:
        stats["avg_text_length"] = stats["total_chars"] // stats["total_texts"]
        stats["size_mb"] = stats["total_bytes"] / 1024 / 1024
    else:
        stats["size_mb"] = 0

    print(f"\n\nDownload complete!")
    print(f"  Texts: {stats['total_texts']}")
    print(f"  Size: {stats['size_mb']:.2f}MB")
    print(f"  Avg length: {stats['avg_text_length']} chars")
    print(f"  Filtered (low score): {stats['texts_filtered']}")
    print(f"  Output: {output_path}")

    # Save stats
    stats_path = output_file.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")

    return stats


def prepare_train_val_split(
    input_path: str,
    train_path: str = "data/fineweb_edu_train.jsonl",
    val_path: str = "data/fineweb_edu_val.jsonl",
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Split a JSONL file into train and validation sets.

    Args:
        input_path: Input JSONL file
        train_path: Output training file
        val_path: Output validation file
        val_ratio: Fraction for validation (default 5%)
        seed: Random seed

    Returns:
        Statistics dictionary
    """
    import random

    print(f"Splitting {input_path} into train/val...")

    # Load all records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(line)

    # Shuffle
    random.seed(seed)
    random.shuffle(records)

    # Split
    split_idx = int(len(records) * (1 - val_ratio))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    # Write
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_records)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_records)

    stats = {
        "total": len(records),
        "train": len(train_records),
        "val": len(val_records),
        "train_path": train_path,
        "val_path": val_path,
    }

    print(f"Split complete:")
    print(f"  Train: {stats['train']} texts -> {train_path}")
    print(f"  Val: {stats['val']} texts -> {val_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu subset for SCU training"
    )
    parser.add_argument(
        "--size", type=float, default=50.0,
        help="Target size in MB (default: 50)"
    )
    parser.add_argument(
        "--output", default="data/fineweb_edu_sample.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--min-score", type=float, default=3.0,
        help="Minimum educational score (0-5, default: 3.0)"
    )
    parser.add_argument(
        "--max-texts", type=int, default=None,
        help="Maximum number of texts to download"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Also create train/val split"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05,
        help="Validation ratio for split (default: 0.05)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Download
    stats = download_fineweb_edu(
        target_size_mb=args.size,
        output_path=args.output,
        min_score=args.min_score,
        max_texts=args.max_texts,
        seed=args.seed,
    )

    # Optionally split
    if args.split:
        base = Path(args.output).stem
        train_path = f"data/{base}_train.jsonl"
        val_path = f"data/{base}_val.jsonl"
        prepare_train_val_split(
            args.output,
            train_path=train_path,
            val_path=val_path,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
