# Repo Map

High-level guide to what lives where.

## Core SCU Code

- `shannon_control/`: canonical SCU implementation (controller, metrics, MLX)
- `scu/`: thin compatibility wrapper for legacy imports
- `scu_api/`: training engine + API + CLI (optional)

## Training and Config

- `scripts/`: entrypoint scripts for training, eval, and experiments
- `configs/`: YAML defaults and templates
- `data/`: datasets (local, not versioned)
- `adapters/`: LoRA adapter outputs
- `logs/`: training logs and CSV metrics

## Research and Artifacts

- `experiments/`, `results/`, `ablations/`: experimental runs and outputs
- `notebooks/`: exploratory notebooks
- `papers/`: writeups and drafts
- `archive/`: older or shelved work

## Supporting Docs and Apps

- `docs/`: guides, technical notes, and references
- `examples/`: usage examples
- `tests/`: tests
- `labs-frontend/`: demo UI / slides
