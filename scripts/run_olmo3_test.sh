#!/bin/bash
# OLMo 3 7B - SCU Quick Test
#
# Hardware support:
#   - RTX 3080 10GB: Uses Unsloth 4-bit quantization
#   - Mac M4 Max 36GB: Uses MLX 4-bit quantization
#
# Usage:
#   ./scripts/run_olmo3_test.sh           # Auto-detect hardware
#   ./scripts/run_olmo3_test.sh cuda      # Force CUDA
#   ./scripts/run_olmo3_test.sh mlx       # Force MLX

set -e

BACKEND=${1:-""}

echo "============================================"
echo "OLMo 3 7B - SCU Test Run"
echo "============================================"
echo ""

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q datasets transformers peft accelerate bitsandbytes

# Check for Unsloth (CUDA) or MLX (Mac)
if [ "$BACKEND" = "cuda" ] || python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA detected, installing Unsloth..."
    pip install -q unsloth 2>/dev/null || echo "Unsloth optional - using standard transformers"
elif [ "$BACKEND" = "mlx" ] || [ "$(uname)" = "Darwin" ]; then
    echo "Mac detected, installing MLX..."
    pip install -q mlx mlx-lm 2>/dev/null || echo "MLX optional"
fi

echo ""
echo "Step 1: Download FineWeb-Edu (~50MB)..."
python scripts/load_fineweb_edu.py --size 50 --output data/fineweb_edu_sample.jsonl

echo ""
echo "Step 2: Run 100 training steps with SCU..."
if [ -n "$BACKEND" ]; then
    python scripts/run_olmo3_test.py --backend "$BACKEND" --steps 100
else
    python scripts/run_olmo3_test.py --steps 100
fi

echo ""
echo "============================================"
echo "Test complete! Check logs/ for training curves."
echo "============================================"
