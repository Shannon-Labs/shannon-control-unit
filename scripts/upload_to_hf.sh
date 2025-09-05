#!/bin/bash

# Upload script for HuggingFace
# Run this after configuring HF MCP server or logging in with huggingface-cli

echo "🚀 Uploading Shannon Control Unit files to HuggingFace..."
echo ""

# Check if logged in
if ! huggingface-cli whoami &>/dev/null && ! hf auth whoami &>/dev/null; then
    echo "❌ Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    echo "Or configure MCP server with:"
    echo "claude mcp add hf-mcp-server -t http https://huggingface.co/mcp -H \"Authorization: Bearer <your_token>\""
    exit 1
fi

REPO="hunterbown/shannon-control-unit"
echo "📦 Uploading to $REPO..."
echo ""

# Upload README
echo "📝 Uploading README..."
huggingface-cli upload $REPO README_HF.md README.md --repo-type model 2>&1 | tail -1

# Upload figures (PNG only)
echo "🖼️  Uploading figures (PNG)..."
for file in assets/figures/*.png; do
    if [ -f "$file" ]; then
      echo "  • Uploading $file..."
      huggingface-cli upload $REPO "$file" "$file" --repo-type model --quiet 2>&1 | tail -1
    fi
done

# Upload demo notebook
echo "📓 Uploading demo notebook..."
if [ -f notebooks/SCU_Demo.ipynb ]; then
  huggingface-cli upload $REPO notebooks/SCU_Demo.ipynb notebooks/SCU_Demo.ipynb --repo-type model --quiet 2>&1 | tail -1
elif [ -f web/shannon_scu_demo.ipynb ]; then
  huggingface-cli upload $REPO web/shannon_scu_demo.ipynb notebooks/SCU_Demo.ipynb --repo-type model --quiet 2>&1 | tail -1
else
  echo "  ⚠️  No demo notebook found (skipping)"
fi

# Upload ablation data
echo "📊 Uploading ablation data..."
for file in ablations/*.csv; do
    echo "  • Uploading $file..."
    huggingface-cli upload $REPO "$file" "$file" --repo-type model --quiet 2>&1 | tail -1
done

# Upload validation artifacts
echo "✅ Uploading validation artifacts..."
huggingface-cli upload $REPO results/3b_validation_results.json results/3b_validation_results.json --repo-type model --quiet 2>&1 | tail -1

# Upload evaluation script
echo "🔬 Uploading evaluation script..."
huggingface-cli upload $REPO scripts/eval_bpt.py scripts/eval_bpt.py --repo-type model --quiet 2>&1 | tail -1

echo ""
echo "✅ Upload complete!"
echo "View at: https://huggingface.co/$REPO"
