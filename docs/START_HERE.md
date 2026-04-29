# Start Here: Training a New Model with SCU

This repo is a mix of research, experiments, and production-ish training code. If
you want to try SCU on a new model, start with LoRA fine-tuning. It is the most
tested path here. Pretraining is possible in theory but not wired end-to-end yet.

## 1) Decide: fine-tune vs pretrain

- Fine-tune (recommended): supported on both PyTorch/CUDA and MLX.
- Pretrain: not fully implemented. It needs full-parameter ParamBPT and a
  pretraining-grade data pipeline. If you want this, we can add a "full" mode.

## 2) Pick your backend

### PyTorch / CUDA (NVIDIA)

Use the generic LoRA script:

```bash
python scripts/train_scu.py \
  --base_model meta-llama/Llama-3.2-1B \
  --train_data data/train.txt \
  --steps 500 \
  --adapter_out adapters/my_scu_run
```

### Apple Silicon / MLX

Use the MLX training script:

```bash
python scripts/train_mlx_scu.py \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train-data data/train.jsonl \
  --steps 500 \
  --adapter-out adapters/my_mlx_run
```

Full-parameter mode (experimental, heavier):

```bash
python scripts/train_mlx_scu.py \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --train-data data/train.jsonl \
  --steps 500 \
  --full-params \
  --adapter-out adapters/my_mlx_full_run
```

## 3) Data format

- `.txt`: plain text, paragraphs separated by blank lines
- `.jsonl`: one JSON object per line with a `text` field

See `shannon_control/data.py` for the PyTorch loader behavior.

FineWiki helper:

```bash
python scripts/download_finewiki.py --target-mb 500 --output data/finewiki_en.jsonl
```

## 4) Where to tune defaults

- `configs/default.yaml`: starting hyperparameters and LoRA defaults
- `configs/default.yaml` (baseline_suite): fixed datasets + seeds for reproducible eval
- `scu_api/service/smart_config.py`: auto-config table by model size
- `shannon_control/control.py`: PI controller implementation

## 5) When to stop

Training is "done" when:

- `lambda` stops changing (stable over a window), and
- `S` is near the target (within ~5%) as an internal diagnostic, and
- loss is flat or rising

Treat S/S* as controller diagnostics; headline outcomes are BPT/PPL,
domain retention, and compute-adjusted improvement.

## 6) Next checks

- `scripts/eval_bpt.py` for BPT/PPL, retention, and PAG (JSON summary output)
- `scripts/eval_quality.py --summary_json <same.json>` to append qualitative samples

If you want pretraining support, tell me your target model + hardware and I can
add a full-parameter training path.
