#!/bin/bash
# VibeThinker 1.5B - Training with Better Data (FineWiki)
# This uses a larger dataset (100MB+) to properly justify the 1.5B model size.

set -e

echo "🔬 VibeThinker 1.5B - Phase 2 Training (Better Data)"
echo "====================================================="

# Configuration
MODEL_ID="$(pwd)/models/VibeThinker-1.5B"
DATA_PATH="data/train_v2.txt"
ADAPTER_OUT="adapters/vibethinker_1.5b_v2"
LOG_FILE="logs/vibethinker_v2_$(date +%Y%m%d_%H%M%S).csv"

# Training parameters
STEPS=1000  # Increased steps for more data
BATCH_SIZE=1
GRAD_ACCUM=16
TARGET_S=0.01
KP=0.8
KI=0.15

# Check prerequisites
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Training data not found at $DATA_PATH"
    echo "   Please run: python scripts/download_finewiki.py"
    exit 1
fi

echo ""
echo "📋 Configuration:"
echo "  Base Model: $MODEL_ID"
echo "  Training Data: $DATA_PATH"
echo "  Output Adapter: $ADAPTER_OUT"
echo "  Log File: $LOG_FILE"
echo ""
echo "  Steps: $STEPS"
echo "  Target S: $TARGET_S (1%)"
echo ""

echo "🚀 Starting training..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run training
python scripts/train_scu.py \
  --base_model "$MODEL_ID" \
  --train_data "$DATA_PATH" \
  --steps $STEPS \
  --batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --target_s $TARGET_S \
  --kp $KP \
  --ki $KI \
  --adapter_out "$ADAPTER_OUT" \
  --log_csv "$LOG_FILE" \
  # --tokens_per_epoch_override 100000000  # Uncomment to force normalization if data is small

echo ""
echo "✅ Training complete!"
echo "📁 Adapter saved to: $ADAPTER_OUT"
