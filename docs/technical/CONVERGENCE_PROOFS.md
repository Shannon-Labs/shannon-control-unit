# Convergence Analysis of the Shannon Control Unit

**Status**: Revised to match implementation (Negative Plant Gain).

## 1. Problem Formulation

Consider the adaptive regularization system with state variables:
- $\lambda(t) \in [\lambda_{\min}, \lambda_{\max}]$: regularization parameter
- $S(t) \in [0,1]$: information ratio
- $I(t) \in [I_{\min}, I_{\max}]$: integral state

The control objective is to track a reference $S^* \in (0,1)$.

## 2. System Dynamics (Corrected)

### 2.1 Plant Model: Negative Gain
The code implements a "negative plant": increasing $\lambda$ strengthens regularization, which suppresses weights and reduces ParamBPT. Thus:
$$ \frac{\partial S}{\partial \lambda} < 0 $$

We approximate the plant locally as:
$$ S_{k+1} \approx S_k + K_{plant} (\lambda_{k+1} - \lambda_k) $$
where $K_{plant} < 0$.

### 2.2 Controller: Positive Exponent
To counteract the negative plant gain, the controller must use **positive feedback** on the error $e = S - S^*$.
- If $S > S^*$ ($e > 0$), we need to **decrease** $S$. Since $\partial S/\partial \lambda < 0$, we must **increase** $\lambda$.
- Therefore, the update rule must be increasing in $e$.

The implemented update is:
$$ \lambda_{k+1} = \lambda_k \cdot \exp(u_k) $$
$$ u_k = K_p e_k + K_i \sum e_\tau $$

This confirms the sign convention: $e > 0 \implies u > 0 \implies \lambda \uparrow \implies S \downarrow$.

## 3. Stability Analysis (Small Signal)

Let $x_k = \ln \lambda_k$. The update becomes additive in log-space:
$$ x_{k+1} = x_k + K_p e_k + K_i I_{k+1} $$

Linearizing the plant around an operating point $(\lambda_0, S_0)$:
$$ S_k \approx S_0 + \frac{\partial S}{\partial \ln \lambda} (x_k - x_0) $$
Let $G = \frac{\partial S}{\partial \ln \lambda} = \lambda \frac{\partial S}{\partial \lambda}$. Since $\frac{\partial S}{\partial \lambda} < 0$ and $\lambda > 0$, we have **$G < 0$**.

Substituting into the error dynamics $e_k = S_k - S^*$:
$$ e_{k+1} - e_k \approx G (x_{k+1} - x_k) $$
$$ e_{k+1} - e_k \approx G (K_p e_k + K_i e_k) \quad \text{(assuming simple integral update)} $$
$$ e_{k+1} \approx e_k (1 + G(K_p + K_i)) $$

### 3.1 Stability Condition
For stability, we need $|1 + G(K_p + K_i)| < 1$.
Since $G < 0$, let $g = -G = |G|$.
$$ |1 - g(K_p + K_i)| < 1 $$
$$ -1 < 1 - g(K_p + K_i) < 1 $$

This implies two conditions:
1. $g(K_p + K_i) > 0 \implies$ True since $g, K_p, K_i > 0$.
2. $g(K_p + K_i) < 2$

**Result**: The system is stable if the total loop gain is less than 2.
$$ (K_p + K_i) < \frac{2}{|G|} $$

### 3.2 Interpretation
The "stiffness" of the system depends on the local sensitivity $|G| = |\lambda \partial S / \partial \lambda|$. If the model is very sensitive to regularization (large $|G|$), gains must be lowered to maintain stability.

## 4. Nonlinearities and Robustness

The actual system includes:
- **Deadband**: Ignores small errors, preventing limit cycles due to noise.
- **Saturation**: $\lambda$ is clipped to $[\lambda_{\min}, \lambda_{\max}]$.
- **Anti-windup**: Integral term is clamped.

These nonlinearities generally improve stability (passivity) but make formal global proof difficult. The small-signal analysis above provides the practical tuning guidelines used in `auto-config`.

## 5. Conclusion

The implemented control law ($\lambda \cdot \exp(+u)$) is consistent with the negative plant physics ($\partial S/\partial \lambda < 0$). Stability is conditional on the loop gain product being $< 2$.
