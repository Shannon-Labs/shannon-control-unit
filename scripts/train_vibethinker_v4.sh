#!/bin/bash
# VibeThinker 1.5B - Phase 4 Training (Adaptive Prior)
# This uses dynamic prior adjustment to keep lambda in healthy range despite rapid learning

set -e

echo "🔬 VibeThinker 1.5B - Phase 4 Training (Adaptive Optimization)"
echo "======================================================="
echo "Goal: Prevent lambda saturation by adapting prior width to learning progress"
echo ""

MODEL_ID="$(pwd)/models/VibeThinker-1.5B"
DATA_PATH="data/train_v2.txt"
ADAPTER_OUT="adapters/vibethinker_1.5b_v4"
LOG_FILE="logs/vibethinker_v4_$(date +%Y%m%d_%H%M%S).csv"

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

echo "🚀 Starting training (Dynamic Prior Mode)"
echo "  Target S: 0.01 (1%)"
echo "  Prior: Dynamic (0.01 -> 0.05 based on learning)"
echo "  Steps: 300 (optimized run)"
echo ""

source .venv/bin/activate

python scripts/train_scu.py \
  --base_model "$MODEL_ID" \
  --train_data "$DATA_PATH" \
  --steps 300 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --target_s 0.01 \
  --kp 0.8 \
  --ki 0.15 \
  --dynamic_prior \
  --prior_sigma_initial 0.01 \
  --prior_sigma_max 0.05 \
  --adapter_out "$ADAPTER_OUT" \
  --log_csv "$LOG_FILE"

echo ""
echo "✅ Training complete!"
echo "📁 Adapter saved to: $ADAPTER_OUT"
echo "📊 See training dynamics: $LOG_FILE"
echo ""
echo "Expected results:"
echo "  - Lambda: Stays in healthy range [0.5, 1.5] throughout"
echo "  - No saturation at end despite rapid learning"
