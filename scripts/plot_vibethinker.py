import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Find the latest log file
log_dir = 'logs'
files = [f for f in os.listdir(log_dir) if f.startswith('vibethinker_validated') and f.endswith('.csv')]
if not files:
    print("No log files found")
    sys.exit(1)
    
latest_file = max([os.path.join(log_dir, f) for f in files], key=os.path.getctime)
print(f"Plotting data from {latest_file}")

df = pd.read_csv(latest_file)

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: S vs Target
ax1.plot(df['step'], df['S'], label='Measured S', color='blue', linewidth=2)
ax1.axhline(y=0.01, color='red', linestyle='--', label='Target S (1%)', linewidth=2)
ax1.set_ylabel('S Ratio')
ax1.set_title('S Control: Measured vs Target')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.0)  # S is 0 to 1

# Plot 2: Lambda
ax2.plot(df['step'], df['lambda'], label='Lambda', color='green', linewidth=2)
ax2.set_ylabel('Lambda')
ax2.set_title('Control Effort (Lambda)')
ax2.grid(True, alpha=0.3)

# Plot 3: BPT Components
ax3.plot(df['step'], df['data_bpt'], label='Data BPT (Prediction Error)', color='orange', alpha=0.7)
ax3.plot(df['step'], df['param_bpt'], label='Param BPT (Complexity)', color='purple', alpha=0.7)
ax3.set_ylabel('Bits Per Token')
ax3.set_xlabel('Step')
ax3.set_title('Information Components')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'assets/figures/vibethinker_training_dynamics.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
