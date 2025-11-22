#!/bin/bash
# OLMo 7B - SCU Validation Run
# Goal: Test if SCU behaves differently on a "Base" model vs. the "Instruct" VibeThinker
# Hypothesis: OLMo 7B will NOT saturate at step 386 because it needs more training to learn the task.

set -e

echo "🧪 OLMo 7B - SCU Validation Run"
echo "================================="
echo ""

# Using the official AllenAI OLMo-7B base model
MODEL_ID="allenai/OLMo-7B"
DATA_PATH="data/train_v2.txt"
ADAPTER_OUT="adapters/olmo_7b_scu"
LOG_FILE="logs/olmo_7b_scu_$(date +%Y%m%d_%H%M%S).csv"

echo "📋 Configuration:"
echo "  Base Model: $MODEL_ID"
echo "  Training Data: $DATA_PATH"
echo "  Output Adapter: $ADAPTER_OUT"
echo "  Log File: $LOG_FILE"
echo ""

source .venv/bin/activate

# We use the exact same SCU parameters as VibeThinker V3 for a fair comparison.
# target_s=0.01, kp=0.8, ki=0.15
# Batch size 1 is CRITICAL for 36GB RAM safety.
# 7B params (FP16) = ~14GB
# Optimizer states + Gradients + Activation overhead = ~6-8GB with LoRA
# Total estimated peak: ~22-24GB. This fits in 36GB comfortably but leaves little room for multitasking.
# 500 steps is enough to see the INITIAL curve. If it doesn't saturate by 500, our hypothesis is confirmed.

# Install required dependency for OLMo architecture
pip install ai2-olmo || echo "ai2-olmo installation failed, check environment"

python scripts/train_scu.py \
  --base_model "$MODEL_ID" \
  --train_data "$DATA_PATH" \
  --steps 500 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --target_s 0.01 \
  --kp 0.8 \
  --ki 0.15 \
  --adapter_out "$ADAPTER_OUT" \
  --log_csv "$LOG_FILE"

echo ""
echo "✅ OLMo Training complete!"
echo "📊 See training dynamics: $LOG_FILE"
