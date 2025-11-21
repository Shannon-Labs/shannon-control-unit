"""Dataset utilities for SCU training."""

import json
import random
from typing import List, Dict, Iterator, Optional
from pathlib import Path


def tokenize_and_chunk(
    texts: List[str],
    tokenizer,
    block_size: int = 4096,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict[str, List[int]]]:
    """Tokenize texts and chunk into fixed-size blocks.
    
    Concatenates all tokenized texts, then slices into fixed windows.
    No ragged tails: last incomplete block is dropped.
    
    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        block_size: Size of each chunk (default 4096)
        shuffle: Whether to shuffle chunks
        seed: Random seed for shuffling
        
    Returns:
        List of dicts with 'input_ids' and 'attention_mask'
        
    # Unit tests:
    >>> # All chunks should be exactly block_size
    >>> # Number of chunks = floor(total_tokens / block_size)
    """
    # Tokenize all texts and concatenate
    all_tokens = []
    for text in texts:
        tokens = tokenizer(text, truncation=False, add_special_tokens=False)
        all_tokens.extend(tokens['input_ids'])
    
    # Chunk into blocks
    chunks = []
    for i in range(0, len(all_tokens) - block_size + 1, block_size):
        chunk_ids = all_tokens[i:i + block_size]
        chunks.append({
            'input_ids': chunk_ids,
            'attention_mask': [1] * len(chunk_ids)
        })
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(chunks)
    
    return chunks


def load_texts_from_file(filepath: str, max_texts: Optional[int] = None) -> List[str]:
    """Load texts from a file.
    
    Supports:
    - .txt: One document (split by double newlines for paragraphs)
    - .jsonl: One JSON object per line with 'text' field
    
    Args:
        filepath: Path to text file
        max_texts: Maximum number of texts to load
        
    Returns:
        List of text strings
    """
    path = Path(filepath)
    texts = []
    
    if path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_texts and len(texts) >= max_texts:
                    break
                data = json.loads(line)
                texts.append(data.get('text', ''))
    else:
        # Treat as plain text
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by double newlines for paragraphs
            paragraphs = content.split('\n\n')
            texts = [p.strip() for p in paragraphs if p.strip()]
            if max_texts:
                texts = texts[:max_texts]
    
    return texts


def create_data_iterator(
    chunks: List[Dict[str, List[int]]],
    batch_size: int = 4
) -> Iterator[List[Dict[str, List[int]]]]:
    """Create a batch iterator over chunks.
    
    Args:
        chunks: List of tokenized chunks
        batch_size: Batch size
        
    Yields:
        Batches of chunks
    """
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]


def estimate_tokens(texts: List[str], chars_per_token: float = 4.0) -> int:
    """Rough estimate of total tokens in texts.
    
    Args:
        texts: List of text strings
        chars_per_token: Average characters per token (default 4.0)
        
    Returns:
        Estimated token count
    """
    total_chars = sum(len(text) for text in texts)
    return int(total_chars / chars_per_token)


def prepare_dataset(
    train_file: str,
    tokenizer,
    block_size: int = 4096,
    val_file: Optional[str] = None,
    val_split: float = 0.1,
    seed: int = 42
) -> tuple:
    """Prepare training and validation datasets.
    
    Args:
        train_file: Path to training data
        tokenizer: HuggingFace tokenizer
        block_size: Chunk size
        val_file: Optional validation file
        val_split: If no val_file, fraction to use for validation
        seed: Random seed
        
    Returns:
        (train_chunks, val_chunks, metadata)
    """
    # Load training texts
    train_texts = load_texts_from_file(train_file)
    
    # Split or load validation
    if val_file:
        val_texts = load_texts_from_file(val_file)
    else:
        # Split from training data
        random.seed(seed)
        random.shuffle(train_texts)
        split_idx = int(len(train_texts) * (1 - val_split))
        val_texts = train_texts[split_idx:]
        train_texts = train_texts[:split_idx]
    
    # Tokenize and chunk
    train_chunks = tokenize_and_chunk(train_texts, tokenizer, block_size, shuffle=True, seed=seed)
    val_chunks = tokenize_and_chunk(val_texts, tokenizer, block_size, shuffle=False, seed=seed)
    
    # Calculate metadata
    metadata = {
        'block_size': block_size,
        'train_chunks': len(train_chunks),
        'val_chunks': len(val_chunks),
        'train_tokens': len(train_chunks) * block_size,
        'val_tokens': len(val_chunks) * block_size,
        'seed': seed
    }
    
    return train_chunks, val_chunks, metadata