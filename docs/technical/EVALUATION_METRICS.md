# Evaluation Metrics for Shannon Control Unit

Hunter Bown — Shannon Labs

Public reporting should focus on generalization, retention, and efficiency.
The information ratio $S$ and target $S^*$ are **internal controller diagnostics**
and should not be used as headline performance metrics.

**Metric Stack (research-facing)**:

| Metric | Why it matters | How to compute in this repo |
| --- | --- | --- |
| BPT / PPL (held-out) | Core generalization for next-token models | `scripts/eval_bpt.py --texts data/fineweb_edu_1gb_val.jsonl` |
| Domain retention (benchmark score) | Catastrophic forgetting signal | `scripts/eval_bpt.py --domain_texts <domain.jsonl>` or pass `--domain_score_*` |
| Generalization gap (train-val BPT) | Overfitting signal | `scripts/eval_bpt.py --train_texts data/fineweb_edu_1gb_train.jsonl` |
| Compute-adjusted improvement | Practical impact per step/FLOP | `scripts/eval_bpt.py --steps <n>` or `--total_flops <n>` |
| Control stability (e_RMS, overshoot, OSC) | Safe control dynamics | `scripts/eval_bpt.py --control_log logs/scu_training.csv --target_s <S*>` |

## 1. External Outcome Metrics (Headline)

### 1.1 Bits-Per-Token (BPT)

**Definition**: Information-theoretic measure of model compression efficiency.

$$\text{BPT} = -\frac{1}{N \ln 2} \sum_{i=1}^N \log p(x_i|\theta)$$

**Properties**:
- Lower is better (higher compression)
- Scale-invariant across vocabulary sizes
- Directly related to cross-entropy loss: BPT = CE / ln(2)
- Theoretical minimum: H(X) = entropy of data distribution

**Measurement Protocol**:
1. Evaluate on held-out test set (minimum 1M tokens)
2. Use teacher forcing (no sampling)
3. Average over sequence dimension
4. Report mean ± standard error

**Observed (this repo):**
| Model | Baseline BPT | SCU BPT | Improvement |
|-------|--------------|---------|-------------|
| 1B | 3.920 | 3.676 | -6.2% |
| 3B | 1.830 | 1.635 | -10.6% |

### 1.2 Perplexity

**Definition**: Exponential of cross-entropy, interpretable as average branching factor.

$$\text{PPL} = \exp(\text{CE}_{\text{nats}}) = 2^{\text{BPT}}$$

**Properties**:
- Range: [1, ∞)
- Lower is better
- Intuition: Average number of equally likely choices
- Non-linear transformation of BPT

**Relationship to BPT**:
$$\Delta\text{PPL} \approx \text{PPL}_{\text{base}} \cdot \ln(2) \cdot \Delta\text{BPT}$$

### 1.3 Domain Retention (Benchmark Score)

**Definition**: A fixed-domain benchmark metric where higher is better
(e.g., accuracy, pass@k, or $1/\text{BPT}$ on a locked domain-held-out set).

$$P = \frac{\text{DomainScore}_{\text{finetuned}}}{\text{DomainScore}_{\text{base}}}$$

**Repo default**: `scripts/eval_bpt.py --domain_texts ...` uses
$$\text{DomainScore} = \frac{1}{\text{BPT}_{\text{domain}}}$$
so $P = \text{BPT}_{\text{base}} / \text{BPT}_{\text{finetuned}}$ on the domain set.

Note: a locked domain benchmark must be selected (math/code/etc.) before reporting retention.

### 1.4 Generalization Gap (Train-Val BPT)

**Definition**: Overfitting signal measured as train minus validation BPT.

$$\text{Gap} = \text{BPT}_{\text{train}} - \text{BPT}_{\text{val}}$$

Lower is better; large positive gaps indicate overfitting.

### 1.5 Preservation-Adjusted Generalization (PAG)

**Definition**: External success metric combining generalization and retention.

Let
$$G = \frac{\text{BPT}_{\text{base}}}{\text{BPT}_{\text{finetuned}}}$$
and
$$P = \frac{\text{DomainScore}_{\text{finetuned}}}{\text{DomainScore}_{\text{base}}}$$

Then
$$\text{PAG} = \sqrt{G \cdot P}$$

PAG > 1 indicates improved generalization with preserved or improved domain skill.

## 2. Control Diagnostics (Internal Only)

### 2.1 Information Ratio (S)

**Definition**: Relative contribution of parameter complexity.

$$S = \frac{\text{ParamBPT}}{\text{DataBPT} + \text{ParamBPT}}$$

**Components**:
- **DataBPT**: From cross-entropy loss
- **ParamBPT**: $\frac{1}{N\ln 2} \sum_j \frac{\theta_j^2}{2\sigma^2}$

**Target Selection**:
- Empirically determined per model size and setup
- Typical range (observed here): ≈1% (1B), ≈2.9% (3B)
- Scaling with model size is a hypothesis to test, not established
- **Note**: $S$ values depend on the normalization constant $N$ (`tokens_per_epoch`). Always report $N$ alongside $S$.
- **Reporting**: Treat $S$ and $S^*$ as internal diagnostics (tracking error, overshoot, OSC), not public success metrics.

### 2.2 Control Error Metrics

**Tracking Error**:
$$e_{\text{RMS}} = \sqrt{\frac{1}{T} \sum_{t=1}^T (S_t - S^*)^2}$$

**Steady-State Error**:
$$e_{ss} = \lim_{t \to \infty} |S_t - S^*|$$

**Maximum Overshoot**:
$$M_p = \max_t \frac{S_t - S^*}{S^*} \times 100\%$$

## 3. Convergence Metrics

### 3.1 Time to Convergence

**Definition**: Steps to reach and maintain target within tolerance.

$$T_c = \min\{t : \forall \tau > t, |S_\tau - S^*| < \epsilon\}$$

where ε = 0.002 (0.2 percentage points)

Qualitative expectations (not measured here):
- Full PI control: reaches band and maintains it
- P-only: faster rise, more oscillation
- I-only: slower convergence, less oscillation

### 3.2 Settling Time

**Definition**: Time to reach and stay within 5% of final value.

$$T_s = \min\{t : \forall \tau > t, |S_\tau - S_{\infty}| < 0.05 S_{\infty}\}$$

### 3.3 Rise Time

**Definition**: Time from 10% to 90% of final value.

$$T_r = t_{90\%} - t_{10\%}$$

## 4. Stability Metrics

### 4.1 Oscillation Index

**Definition**: Measure of control signal variability.

$$\text{OSC} = \frac{\sigma(\lambda)}{\bar{\lambda}} \times 100\%$$

**Interpretation**:
- < 5%: Very stable
- 5-15%: Acceptable
- > 15%: Excessive oscillation

### 4.2 Phase Margin

**Definition**: Distance from instability in frequency domain.

$$\phi_m = 180° + \angle G(j\omega_{gc})$$

where ω_gc is the gain crossover frequency.

**Target**: φ_m > 45° for robust stability

### 4.3 Lyapunov Exponent

**Definition**: Rate of convergence/divergence.

$$\lambda_L = \lim_{t \to \infty} \frac{1}{t} \ln \frac{|\delta S(t)|}{|\delta S(0)|}$$

**Interpretation**:
- λ_L < 0: Stable (converges)
- λ_L = 0: Marginally stable
- λ_L > 0: Unstable (diverges)

## 5. Efficiency Metrics

### 5.1 Computational Overhead

**Wall-Clock Time**:
$$\text{Overhead}_{\text{time}} = \frac{T_{\text{SCU}} - T_{\text{baseline}}}{T_{\text{baseline}}} \times 100\%$$

**Memory Usage**:
$$\text{Overhead}_{\text{memory}} = \frac{M_{\text{SCU}} - M_{\text{baseline}}}{M_{\text{baseline}}} \times 100\%$$

**Target**: < 1% for both metrics

### 5.2 Compute-Adjusted Improvement

Define improvement per unit of compute:

$$\text{Efficiency}_{\text{steps}} = \frac{\Delta\text{BPT}}{\text{Steps}}$$
$$\text{Efficiency}_{\text{FLOPs}} = \frac{\Delta\text{BPT}}{\text{FLOPs}}$$

Use steps when FLOPs are not available; always report the denominator used.

## 6. Generalization Metrics

### 6.1 Domain Transfer

**Definition**: Performance on out-of-domain data.

$$\text{Transfer} = \frac{\text{BPT}_{\text{OOD}}}{\text{BPT}_{\text{ID}}}$$

We have not yet reported cross-domain transfer results in this repo.

### 6.2 Robustness Score

**Definition**: Performance under perturbations.

$$R = 1 - \frac{\max_{\epsilon} |\text{BPT}(\epsilon) - \text{BPT}(0)|}{\text{BPT}(0)}$$

where ε represents various perturbations.

## 7. Statistical Metrics

### 7.1 Confidence Intervals

**Bootstrap 95% CI**:
```python
def bootstrap_ci(metrics, B=10000):
    boots = [np.mean(resample(metrics)) for _ in range(B)]
    return np.percentile(boots, [2.5, 97.5])
```

**Reporting Format**:
BPT = 3.676 [3.651, 3.701]

### 7.2 Effect Size

**Cohen's d**:
$$d = \frac{\mu_{\text{baseline}} - \mu_{\text{SCU}}}{\sigma_{\text{pooled}}}$$

**Interpretation**:
- |d| < 0.2: Negligible
- |d| < 0.5: Small
- |d| < 0.8: Medium
- |d| ≥ 0.8: Large

### 7.3 Statistical Power

**Sample Size Calculation**:
$$n = 2 \left(\frac{(z_{\alpha/2} + z_\beta) \sigma}{\delta}\right)^2$$

For 80% power, α=0.05, δ=0.1 BPT: n ≈ 64

## 8. Visualization Standards

### 8.1 Time Series Plots

**Required Elements**:
- S(t) trajectory with target line
- λ(t) on secondary axis
- Shaded confidence bands
- Convergence time marker

### 8.2 Comparison Plots

**Required Elements**:
- Baseline and SCU BPT over training
- Error bars (standard error)
- Statistical significance markers
- Effect size annotations

### 8.3 Ablation Heatmaps

**Required Elements**:
- Component combinations on axes
- BPT values as colors
- Optimal configuration highlighted
- Interaction effects annotated

## 9. Reporting Standards

### 9.1 Minimal Reporting Set

1. **Performance**: BPT, Perplexity (mean ± SE)
2. **Retention**: DomainScore, retention ratio $P$, PAG
3. **Overfitting**: Generalization gap (train-val BPT)
4. **Efficiency**: Compute-adjusted improvement (per step or FLOP)
5. **Control (internal)**: $S$, $e_{\text{RMS}}$, overshoot, OSC
6. **Statistical**: p-value, effect size, 95% CI
7. **Configuration**: All hyperparameters

### 9.2 Extended Reporting Set

Additionally include:
- Full training curves
- Ablation results
- Robustness analysis
- Cross-domain evaluation
- Statistical power analysis

### 9.3 Reproducibility Checklist

- [ ] Random seeds specified
- [ ] Hardware details documented
- [ ] Software versions listed
- [ ] Dataset splits defined
- [ ] Evaluation code available
- [ ] Raw results provided

## 10. Benchmark Comparisons

We do not include benchmark leaderboards or SOTA comparisons here; only repo-measured 1B/3B values.

## 11. Failure Mode Analysis

### 11.1 Detectable Failures

| Failure Mode | Detection Metric | Threshold | Action |
|--------------|------------------|-----------|--------|
| Divergence | |dS/dt| | > 0.01 | Reset controller |
| Oscillation | OSC | > 20% | Reduce gains |
| Saturation | λ at bounds | > 50% time | Adjust S* |
| Slow convergence | T_c | > 500 steps | Increase Kp |

### 11.2 Quality Assurance

**Automated Checks**:
```python
def validate_training(metrics):
    assert metrics['final_bpt'] < baseline_bpt
    assert metrics['convergence_time'] < 500
    assert metrics['oscillation'] < 0.20
    assert metrics['overhead'] < 0.01
    return True
```

---

*Evaluation Protocol v1.0*
