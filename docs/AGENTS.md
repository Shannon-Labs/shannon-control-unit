# AI Assistant Guide for SCU

This guide is for any AI assistant (Claude, Gemini, GPT, etc.) helping with Shannon Control Unit development.

---

## One-Paragraph Summary

SCU applies **PI control** to LLM fine-tuning regularization. It maintains an optimal **Information Ratio (S = ParamBPT / (DataBPT + ParamBPT))** by adjusting regularization strength λ in real-time. The key discovery is that **when λ stabilizes at equilibrium, training is complete** - the model has learned all meaningful patterns (MDL saturation). This provides provable, information-theoretic stopping criteria instead of arbitrary step counts.

---

## Core Equation

```
S(t) = ParamBPT(t) / (DataBPT(t) + ParamBPT(t))

e(t) = S(t) - S*

λ(t+1) = λ(t) × exp(Kp × e(t) + Ki × Σe(τ))
```

Where:
- **S**: Information ratio (controlled variable)
- **S***: Target ratio (typically 2-3% for LoRA)
- **e**: Error from target
- **λ**: Regularization strength (manipulated variable)
- **Kp, Ki**: Controller gains

Plant gain is negative (∂S/∂λ < 0): increasing λ reduces ParamBPT, which decreases S.

---

## Key Files by Task

| Question | Files to Check |
|----------|----------------|
| "How does SCU work?" | `shannon_control/control.py`, `docs/technical/THEORY.md` |
| "How do I train?" | `scripts/train_olmo3_7b_fineweb.py`, `docs/guides/SCALING_GUIDE.md` |
| "What hyperparameters?" | `scu_api/service/smart_config.py` (CONFIG_SCALES), `configs/default.yaml` |
| "Training not converging?" | Check lambda in logs, verify not [0,0] bounds |
| "When to stop?" | Lambda stable + S near target → done |

---

## Common Modifications Table

| Task | File | What to Change |
|------|------|----------------|
| Add new model size | `scu_api/service/smart_config.py` | Add to `CONFIG_SCALES` dict |
| Change LoRA config | `configs/*.yaml` | `lora_rank`, `lora_alpha`, `lora_targets` |
| Adjust controller | `shannon_control/control.py` | Kp, Ki, deadband, clamp values |
| Add stopping criterion | `shannon_control/mlx/callback.py` | `should_stop_training()` logic |

---

## Terminology Glossary

| Term | Meaning |
|------|---------|
| **BPT** | Bits-Per-Token (information measure) |
| **DataBPT** | Cross-entropy loss converted to bits |
| **ParamBPT** | Parameter complexity in bits (uses prior σ) |
| **S-ratio** | Information ratio = ParamBPT / (DataBPT + ParamBPT) |
| **S*** | Target S-ratio (setpoint) |
| **Lambda (λ)** | Regularization strength (actuator) |
| **Kp** | Proportional gain (responsiveness) |
| **Ki** | Integral gain (steady-state correction) |
| **Deadband** | No-update zone around S* (prevents chatter) |
| **Anti-windup** | Integral clamping to prevent saturation |
| **MDL** | Minimum Description Length principle |
| **Equilibrium** | Lambda stable, S near target = training complete |

---

## Testing Your Changes

### 1. Quick Sanity Check
```bash
python -c "from shannon_control import SCUController; print('OK')"
```

### 2. Run Existing Tests
```bash
pytest tests/ -v
```

### 3. Training Smoke Test
```bash
python scripts/train_olmo3_7b_fineweb.py --steps 10 --adapter-out /tmp/test
# Check: Lambda should change, S-ratio should be computed
```

### 4. Verify Controller is Active
In training output, check:
- Lambda is NOT stuck at 0.0
- Lambda changes over steps
- S-ratio is being computed and logged

---

## Red Flags to Watch For

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| lambda always 0.0 | Lambda bounds [0,0] | Set `--lambda-min 0.0001 --lambda-max 2.0` |
| S-ratio stuck at 0% | ParamBPT not computed | Check `tokens_per_epoch` is set |
| Lambda oscillating wildly | Kp too high | Reduce Kp (try 0.5-0.6) |
| Lambda immediately saturates | S* too aggressive | Increase target_s |
| Loss increasing after equilibrium | Overtraining | Stop at equilibrium point |

---

## Understanding the Control Loop

```
┌─────────────────────────────────────────────────────────────┐
│                        Training Loop                         │
├─────────────────────────────────────────────────────────────┤
│  Data → Model → Loss                                         │
│           │                                                  │
│           ▼                                                  │
│       DataBPT = loss / ln(2)                                │
│       ParamBPT = Σ(w²) / (2σ²·N·ln(2))                      │
│           │                                                  │
│           ▼                                                  │
│       S = ParamBPT / (DataBPT + ParamBPT)                   │
│           │                                                  │
│           ▼                                                  │
│       error = S - S*                                         │
│           │                                                  │
│           ▼                                                  │
│    ┌──────────────────────┐                                 │
│    │    PI Controller     │                                 │
│    │  λ = λ·exp(Kp·e+Ki·I)│                                 │
│    └──────────────────────┘                                 │
│           │                                                  │
│           ▼                                                  │
│       L_total = L_data + λ·L_reg                            │
│           │                                                  │
│           ▼                                                  │
│       Gradient Update                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Helping Humans Understand

When explaining SCU to users:

1. **Start with the problem**: "How do you know when to stop training?"

2. **Explain the insight**: "SCU treats training as a control problem. It monitors how much of the total information is model complexity vs. data fit."

3. **The key result**: "When the controller stabilizes (lambda stops changing), the model has learned everything useful from the data."

4. **Why it matters**: "No more guessing step counts. The math tells you when you're done."

---

**Repository**: https://github.com/Shannon-Labs/shannon-control-unit
**Contact**: hunter@shannonlabs.dev
