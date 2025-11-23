#!/bin/bash
echo "🔄 VibeThinker V3 Training Monitor (500MB, No Override)"
echo "======================================================="
echo ""
echo "Status every 15 seconds (Ctrl+C to stop):"
echo ""

while true; do
    python -c "
import pandas as pd
import glob
import os

# Find the latest V3 CSV
csv_files = glob.glob('logs/vibethinker_v3_*.csv')
if not csv_files:
    print('No V3 training logs yet...')
    exit()

csv_file = sorted(csv_files)[-1]
df = pd.read_csv(csv_file)

if len(df) == 0:
    print('Training starting...')
    exit()

recent = df.tail(20)
print(f'Time: \$(date +%H:%M:%S)')
print(f'CSV: {os.path.basename(csv_file)}')
print(f'Step: {df['step'].iloc[-1]:3.0f}/500 ({df['step'].iloc[-1]/5.0:.1f}%)')
print(f'S:    {df['S'].iloc[-1]:.2%} (last ~20: {recent['S'].mean():.2%}±{recent['S'].std():.3f})')
print(f'λ:    {df['lambda'].iloc[-1]:.3f}')
print(f'Data: {df['data_bpt'].iloc[-1]:.2f}')
print(f'ETA:  ~{(500-len(df))*5/60:.0f} min')
    " 2>/dev/null
    sleep 15
done