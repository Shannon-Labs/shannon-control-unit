# Research Roadmap

**Reporting standard:** Unless noted, report mean $\pm$ 95% CI over $\ge 3$ seeds and paired comparisons versus tuned fixed-$\lambda$ baselines.

## Phase 1: Cross-Architecture Validation (Qwen)

**Objective:** Validate that the Shannon Control Unit (SCU) gains generalize beyond the Llama architecture.

We have observed strong results on Llama 3.2 (1B/3B). The next logical step is to test SCU on **Qwen 2.5** models, which have different architectural biases and pretraining data distributions.

### Planned Experiments
1.  **Baseline Establishment:**
    *   Train Qwen-2.5-1.5B and Qwen-2.5-3B on the WikiText-103 subset using standard Cross-Entropy loss.
    *   Perform a hyperparameter sweep for optimal fixed weight decay/regularization.
    *   **Metric:** Report mean BPT $\pm$ 95% CI over $\ge 3$ seeds.
2.  **SCU Integration:**
    *   Apply SCU (PI Control) to the Qwen training loop.
    *   Target the same Information Ratio ($S^*$) ranges found effective for Llama (approx 1-3%) to test transferability.
3.  **Hypothesis:**
    *   If SCU is a fundamental control mechanism, it should yield similar BPT improvements on Qwen without requiring extensive retuning of the PI gains ($K_p, K_i$).
    *   **Validation:** Paired t-test vs. baseline.

## Phase 2: The Entropy Connection (SCU vs. EntroPIC)

**Objective:** Investigate the theoretical and empirical links between SCU's *Information Ratio* control and EntroPIC's *Entropy* control.

**Context:**
*   **SCU** regulates the trade-off between data fit and model complexity ($S = \frac{\text{ParamBPT}}{\text{TotalBPT}}$).
*   **EntroPIC** regulates the policy entropy ($H(\pi)$) to prevent collapse during RL.

### Research Questions
1.  **Does stabilizing $S$ implicitly stabilize Entropy?**
    *   Hypothesis: maintaining a constant Information Ratio prevents the model from becoming over-confident (low entropy) on noisy data, serving a similar purpose to EntroPIC but derived from MDL principles.
2.  **Convergent Evolution:**
    *   Can we formulate a unified control law that accounts for both?
    *   Is "Information Ratio" simply a more generalized form of "Entropy" when viewed through the lens of compression?

### Experimental Plan
1.  **Instrumentation:**
    *   Add entropy logging to the SCU training loop (monitor average token entropy per batch).
2.  **Correlation Analysis:**
    *   Quantify correlation between SCU control signal $\lambda$ and the measured entropy of the model (Pearson $r$, $p$-value).
    *   *Prediction:* When entropy drops too low (overfitting), SCU should increase $\lambda$ to penalize complexity, thereby raising entropy.
3.  **Direct Comparison:**
    *   Implement an "EntroPIC-style" controller that targets fixed entropy.
    *   Compare the BPT performance of $S$-control vs. $H$-control on the pretraining task.
    *   Ablate by substituting entropy-target controller.

## Phase 3: Scaling Laws for $S^*$

**Objective:** Eliminate the need to manually select the target Information Ratio $S^*$.

*   **Current State:** $S^*$ is a hyperparameter (e.g., 1.0% for 1B, 2.8% for 3B).
*   **Goal:** Discover the function $S^* = f(N, D)$ where $N$ is parameter count and $D$ is dataset size.
*   **Method:**
    *   Run sweeps of $S^*$ across varying model sizes (TinyLlama, Llama-1B, Llama-3B).
    *   Fit a curve to the optimal $S^*$ values to derive a predictive scaling law.
