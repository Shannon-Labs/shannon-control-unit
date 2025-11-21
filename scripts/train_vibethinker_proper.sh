#!/bin/bash
# Proper VibeThinker 1.5B training with full logging and validation
# This addresses the gaps identified in CRITICAL_ANALYSIS.md

set -e

echo "🔬 VibeThinker 1.5B - Proper Training Run with Validation"
echo "=========================================================="

# Configuration
MODEL_ID="$(pwd)/models/VibeThinker-1.5B"
DATA_PATH="data/train.txt"
ADAPTER_OUT="adapters/vibethinker_1.5b_validated"
LOG_FILE="logs/vibethinker_validated_$(date +%Y%m%d_%H%M%S).csv"

# Training parameters (increased from original 100 steps)
STEPS=500
BATCH_SIZE=1
GRAD_ACCUM=16
TARGET_S=0.01
KP=0.8
KI=0.15

# Check prerequisites
if [ ! -d "$MODEL_ID" ]; then
    echo "❌ Error: Base model not found at $MODEL_ID"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Training data not found at $DATA_PATH"
    exit 1
fi

echo ""
echo "📋 Configuration:"
echo "  Base Model: $MODEL_ID"
echo "  Training Data: $DATA_PATH"
echo "  Output Adapter: $ADAPTER_OUT"
echo "  Log File: $LOG_FILE"
echo ""
echo "  Steps: $STEPS (5x original)"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"
echo ""
echo "  Target S: $TARGET_S (1%)"
echo "  Kp: $KP"
echo "  Ki: $KI"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🚀 Starting training..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Run training with proper logging
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
  --log_csv "$LOG_FILE"

echo ""
echo "✅ Training complete!"
echo ""
echo "📊 Next steps:"
echo "  1. Check training logs in: $LOG_FILE"
echo "  2. Plot S convergence and lambda evolution"
echo "  3. Evaluate adapter on validation set"
echo "  4. Compare with non-SCU baseline"
echo ""
echo "📁 Adapter saved to: $ADAPTER_OUT"
