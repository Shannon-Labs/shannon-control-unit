# SCU Control Math Summary

This document summarizes the **actual** control logic implemented in `shannon_control/control.py`.

## 1. Core Definitions

### Information Ratio (S)
The controlled variable is the ratio of parameter information to total information:
$$ S = \frac{\text{ParamBPT}}{\text{DataBPT} + \text{ParamBPT}} $$

- **DataBPT**: Bits-per-token of the cross-entropy loss.
- **ParamBPT**: Bits-per-token of the parameter update cost.
  $$ \text{ParamBPT} = \frac{1}{N \ln 2} \sum \frac{w^2}{2\sigma^2} $$
  *Note: $N$ (`tokens_per_epoch`) is a fixed normalization constant required for reproducibility.*

## 2. Control Loop

### Error Signal
$$ e(t) = S(t) - S^* $$
where $S^*$ is the target setpoint (e.g., 0.01).

### Plant Dynamics (Negative Gain)
The system acts as a "negative plant":
- **Increase $\lambda$** $\to$ Stronger regularization $\to$ Smaller weights $\to$ Lower ParamBPT $\to$ **Decrease $S$**.
- Therefore, $\frac{\partial S}{\partial \lambda} < 0$.

### Controller Update (Positive Feedback)
To regulate a negative plant, the controller uses a positive exponent update:
$$ \lambda_{new} = \lambda_{old} \cdot \exp(u) $$
$$ u = K_p e + K_i \int e \, dt $$

**Logic**:
- If $S > S^*$ ($e > 0$), we need to decrease $S$.
- To decrease $S$, we must increase $\lambda$.
- Since $e > 0$, the term $\exp(u)$ is $> 1$, so $\lambda$ increases. Correct.

## 3. Implementation Details

### Discrete Update
Updates occur at discrete steps $k$:
$$ \lambda_{k+1} = \text{clip}(\lambda_k \cdot \exp(K_p e_k + I_{k+1}), \lambda_{\min}, \lambda_{\max}) $$

### Integral Anti-Windup
$$ I_{k+1} = \text{clip}(I_k + K_i e_k, I_{\min}, I_{\max}) $$
*Integration is paused if $\lambda$ is saturated and the error would push it further into saturation.*

### Deadband
If $|e_k| < \delta$, the error is treated as 0 to prevent chatter.
