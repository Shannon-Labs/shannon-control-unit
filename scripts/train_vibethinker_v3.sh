#!/bin/bash
# VibeThinker 1.5B - Phase 3 Training (500MB, No Override)
# This uses sufficient data for natural S=1% convergence without token count override

set -e

echo "🔬 VibeThinker 1.5B - Phase 3 Training (Scientific Run)"
echo "======================================================="
echo ""

MODEL_ID="$(pwd)/models/VibeThinker-1.5B"
DATA_PATH="data/train_v2.txt"
ADAPTER_OUT="adapters/vibethinker_1.5b_v3"
LOG_FILE="logs/vibethinker_v3_$(date +%Y%m%d_%H%M%S).csv"

echo "📋 Configuration:"
echo "  Base Model: $MODEL_ID"
echo "  Training Data: $DATA_PATH"
echo "  Output Adapter: $ADAPTER_OUT"
echo "  Log File: $LOG_FILE"
echo ""

# Check data size
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Training data not found at $DATA_PATH"
    exit 1
fi

DATA_SIZE=$(ls -lh "$DATA_PATH" | awk '{print $5}')
echo "📊 Data size: $DATA_SIZE"
echo ""

echo "🚀 Starting training (NO override - natural convergence)"
echo "  Target S: 0.01 (1%)"
echo "  Controller: kp=0.8, ki=0.15"
echo "  NO tokens_per_epoch_override - using natural data size"
echo ""

source .venv/bin/activate

python scripts/train_scu.py \
  --base_model "$MODEL_ID" \
  --train_data "$DATA_PATH" \
  --steps 500 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --target_s 0.01 \
  --kp 0.8 \
  --ki 0.15 \
  --adapter_out "$ADAPTER_OUT" \
  --log_csv "$LOG_FILE"

echo ""
echo "✅ Training complete!"
echo "📁 Adapter saved to: $ADAPTER_OUT"
echo "📊 See training dynamics: $LOG_FILE"
echo ""
echo "Expected results:"
echo "  - S ratio: ~0.85% (naturally below 1% target)"
echo "  - Lambda: 0.3-1.0 (active regulation, not saturated)"
echo "  - ParamBPT: ~0.057 (reduced vs 0.075 in V2)"