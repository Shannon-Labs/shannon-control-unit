# VibeThinker 1.5B: Current Run & Next Round

## 📊 Current Run (V2): Worth Continuing ✅

**Status**: Step 526/1000, ~50 minutes remaining  
**Decision**: Continue to completion

Why? Stable convergence, S at target, only 50 min left.

Monitor:
```bash
python -c "
import pandas as pd
df = pd.read_csv('logs/vibethinker_v2_override_20251121_074909.csv')
print(f'Step {df['step'].iloc[-1]:.0f}/1000 | S={df['S'].iloc[-1]:.2%}')
"
```

**Expected**: Finishes in ~50 min, saves adapter to `adapters/vibethinker_1.5b_v2/`

---

## 🎯 Next Round (V3): 500MB Scientific Training

**Trigger**: Start immediately after V2 completes

### Step 1: Get 500MB Data (10-15 min)

```bash
cd /Volumes/VIXinSSD/shannon-control-unit

# Edit download script
nano scripts/download_finewiki.py
# Change line 8: TARGET_SIZE_MB = 500

# Download
source .venv/bin/activate
python scripts/download_finewiki.py
```

Why 500MB? 5× more data = natural S≈0.85% (no override needed).

### Step 2: Train (75 min)

```bash
cat > scripts/train_vibethinker_v3.sh << 'EOF'
#!/bin/bash
cd /Volumes/VIXinSSD/shannon-control-unit
source .venv/bin/activate
python scripts/train_scu.py \
  --base_model models/VibeThinker-1.5B \
  --train_data data/train_v2.txt \
  --steps 500 --batch_size 2 --gradient_accumulation_steps 8 \
  --target_s 0.01 --kp 0.8 --ki 0.15 \
  --adapter_out adapters/vibethinker_1.5b_v3 \
  --log_csv logs/vibethinker_v3_$(date +%Y%m%d_%H%M%S).csv
EOF

bash scripts/train_vibethinker_v3.sh
```

Key changes: No override, batch=2 (faster), 500 steps (sufficient).

**Expected**: S≈0.85%, λ≈0.3-1.0 (healthy, not maxed)

### Step 3: Validate (5 min)

```bash
# Check adapter
ls -lh adapters/vibethinker_1.5b_v3/

# Test inference (when available)
python scripts/test_inference.py \
  --model-path adapters/vibethinker_1.5b_v3 \
  --prompt "Machine learning is"
```

---

## 📈 Comparison

| Round | Data | Tokens | Override | Expected S | Expected λ | Time |
|-------|------|--------|----------|------------|------------|------|
| **V2 (current)** | 100MB | 26M | Yes (100M) | 1.1% | 2.0 (saturated) | 90 min |
| **V3 (next)** | 500MB | 130M | No | 0.85% | 0.3-1.0 (healthy) | 75 min |

**V3 improvement**: More data + no override = healthier controller, mathematically rigorous.

---

## ⏱️ Timeline

```
T+0:   Current V2 at step 526 (monitor)
T+50:  V2 completes → adapter saved
T+50-65: Download 500MB
T+65-140: Train V3
T+140-145: Validate

Total from now: ~145 minutes (2.4 hours)
```

---

## Quick Commands Reference

```bash
# Monitor V2 (every ~10 min)
python -c "import pandas as pd; df=pd.read_csv('logs/vibethinker_v2_override_20251121_074909.csv'); print(f'Step {df['step'].iloc[-1]:.0f}/1000 | S={df['S'].iloc[-1]:.2%}')"

# After V2 completes, download V3 data
nano scripts/download_finewiki.py  # Change TARGET_SIZE_MB=500
python scripts/download_finewiki.py

# Train V3 (use script above)
bash scripts/train_vibethinker_v3.sh

# Monitor V3
python -c "import pandas as pd, glob; df=pd.read_csv(glob.glob('logs/vibethinker_v3_*.csv')[-1]); print(f'Step {df['step'].iloc[-1]:.0f}/500 | S={df['S'].iloc[-1]:.2%} | λ={df['lambda'].iloc[-1]:.3f}')"
```

---

## 🎯 Summary

**Now**: V2 running, ~50 min left, stable, let it finish  
**Next**: 500MB data, no override, healthier training (90 min)  
**Total**: ~140 minutes from now  
**Result**: Mathematically rigorous SCU demonstration

The V3 round will show SCU working naturally without workarounds, providing stronger validation.