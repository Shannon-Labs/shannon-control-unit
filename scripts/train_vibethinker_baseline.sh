#!/bin/bash
# VibeThinker 1.5B - Baseline Training (Standard Finetuning)
# This runs standard Cross-Entropy training (Lambda=0) for comparison

set -e

echo "📉 VibeThinker 1.5B - Baseline Training (Standard Finetuning)"
echo "======================================================="
echo "Goal: Establish baseline performance without SCU regularization"
echo ""

MODEL_ID="$(pwd)/models/VibeThinker-1.5B"
DATA_PATH="data/train_v2.txt"
ADAPTER_OUT="adapters/vibethinker_1.5b_baseline"
LOG_FILE="logs/vibethinker_baseline_$(date +%Y%m%d_%H%M%S).csv"

echo "📋 Configuration:"
echo "  Base Model: $MODEL_ID"
echo "  Training Data: $DATA_PATH"
echo "  Output Adapter: $ADAPTER_OUT"
echo "  Log File: $LOG_FILE"
echo ""

source .venv/bin/activate

# We effectively disable SCU by setting lambda to 0 and locking it there
python scripts/train_scu.py \
  --base_model "$MODEL_ID" \
  --train_data "$DATA_PATH" \
  --steps 500 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --target_s 0.01 \
  --kp 0.0 \
  --ki 0.0 \
  --lambda_init 0.0 \
  --lambda_min 0.0 \
  --lambda_max 0.0 \
  --adapter_out "$ADAPTER_OUT" \
  --log_csv "$LOG_FILE"

echo ""
echo "✅ Baseline Training complete!"
echo "📁 Adapter saved to: $ADAPTER_OUT"
