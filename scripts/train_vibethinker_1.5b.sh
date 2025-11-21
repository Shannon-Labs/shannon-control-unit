#!/bin/bash
# Robust training script for 1.5B model (Qwen/Qwen1.5-1.5B)
# Designed to be "crash-proof" by using conservative memory settings.

set -e

# Default to VibeThinker 1.5B (downloaded locally)
MODEL_ID="$(pwd)/models/VibeThinker-1.5B"
DATA_PATH="data/train.txt" 
ADAPTER_OUT="adapters/vibethinker_1.5b"

# Check if data exists, if not create dummy
if [ ! -f "$DATA_PATH" ]; then
    echo "Creating dummy training data at $DATA_PATH..."
    mkdir -p data
    echo "This is some dummy training data for the vibethinker model." > "$DATA_PATH"
fi

echo "Starting robust training for $MODEL_ID..."
echo "Output adapter: $ADAPTER_OUT"

# Note: We do NOT use --use-unsloth by default to ensure compatibility with Mac/MPS.
# If you are on CUDA, you can add --use-unsloth manually.
# We use batch-size 1 and gradient accumulation to save memory.

scu train \
  --base-model "$MODEL_ID" \
  --train-data "$DATA_PATH" \
  --steps 100 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --fp16 \
  --target-s 0.01 \
  --adapter-out "$ADAPTER_OUT" \
  --wait

echo "Training complete! Adapter saved to $ADAPTER_OUT"
