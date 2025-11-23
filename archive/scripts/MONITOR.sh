#!/bin/bash
echo "🔄 VibeThinker 1.5B V2 Training Monitor"
echo "========================================"
echo ""
echo "Monitoring: logs/vibethinker_v2_override_20251121_074909.csv"
echo ""
echo "Running live updates (Ctrl+C to stop):"
echo ""

while true; do
    python -c "
import pandas as pd
df = pd.read_csv('logs/vibethinker_v2_override_20251121_074909.csv')
print(f'Time: \$(date +%H:%M:%S) | Step: {df['step'].iloc[-1]:4.0f}/1000 | S: {df['S'].iloc[-1]:.2%} | λ: {df['lambda'].iloc[-1]:.3f} | DataBPT: {df['data_bpt'].iloc[-1]:.2f}')
    " 2>/dev/null
    sleep 15
done