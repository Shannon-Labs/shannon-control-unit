# Scripts Index

This folder is a grab-bag of training, evaluation, and research scripts.
Start with the items below and treat the rest as experimental unless noted.

## Training (recommended entrypoints)

- `scripts/train_scu.py`: generic PyTorch LoRA fine-tuning with SCU
- `scripts/train_mlx_scu.py`: MLX LoRA fine-tuning on Apple Silicon
- `scripts/train_olmo3_7b_fineweb.py`: MLX OLMo 3 7B + FineWeb-Edu run
- `scripts/train_olmo_7b_scu.py`: PyTorch OLMo 7B variant

## Evaluation

- `scripts/eval_bpt.py`: compute BPT metrics
- `scripts/eval_quality.py`: quick quality checks
- `scripts/eval_olmo3.py`: OLMo 3 evaluation helper

## Data prep

- `scripts/load_fineweb_edu.py`: download FineWeb-Edu subset
- `scripts/download_finewiki.py`: download FineWiki
- `scripts/split_data.py`: train/val split helper

## Analysis / utilities

- `scripts/compare_models.py`: compare adapter/model outputs
- `scripts/compare_controllers.py`: controller comparison
- `scripts/plot_control.py`: control plots
