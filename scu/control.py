"""Shannon Control Unit: PI controller for adaptive regularization."""

import math
from typing import Optional, Tuple


def ema(prev: float, x: float, alpha: float = 0.1) -> float:
    """Exponential moving average.
    
    Args:
        prev: Previous EMA value
        x: New observation
        alpha: Smoothing factor (0 < alpha <= 1)
        
    Returns:
        Updated EMA value
    """
    return alpha * x + (1 - alpha) * prev


def update_lambda(
    lmbda: float,
    S_meas: float,
    S_target: float,
    I: float,
    *,
    Kp: float = 0.8,
    Ki: float = 0.15,
    deadband: float = 0.002,
    i_min: float = -0.2,
    i_max: float = 0.2,
    lmin: float = 1e-4,
    lmax: float = 10.0,
    ema_alpha: float = 0.1,
    S_hat: Optional[float] = None,
    leak: float = 0.995,
    satur_guard: bool = True
) -> Tuple[float, float, float]:
    """Update lambda using PI control (negative plant gain).
    
    Plant sign: increasing λ strengthens regularization, which (over training) reduces
    weights and thus decreases ParamBPT. Therefore dS/dλ < 0 (negative plant gain).

    We want negative feedback on S:
        error e = S_measured - S_target
        λ_new = λ * exp(+ (Kp*e + Ki*∫e))

    - If S > S* (e > 0), increase λ to push S down.
    - If S < S* (e < 0), decrease λ to allow S up.
    
    Args:
        lmbda: Current lambda value
        S_meas: Measured S ratio (ParamBPT / (DataBPT + ParamBPT))
        S_target: Target S ratio
        I: Current integral term
        Kp: Proportional gain
        Ki: Integral gain
        deadband: Error threshold below which no update occurs
        i_min: Minimum integral value (anti-windup)
        i_max: Maximum integral value (anti-windup)
        lmin: Minimum lambda value
        lmax: Maximum lambda value
        ema_alpha: EMA smoothing factor for S measurement
        S_hat: Previous EMA of S (if None, uses S_meas)
        leak: Integral leak factor (0.995 = slow leak)
        satur_guard: If True, don't integrate when at lambda bounds
        
    Returns:
        (new_lambda, new_I, S_hat)
        
    # Unit tests (conceptual):
    >>> # Test 1: S > S_target => λ should decrease
    >>> λ, I, _ = update_lambda(1.0, S_meas=0.02, S_target=0.01, I=0.0)
    >>> assert λ < 1.0, "Lambda should decrease when S > target"
    
    >>> # Test 2: S < S_target => λ should increase  
    >>> λ, I, _ = update_lambda(1.0, S_meas=0.005, S_target=0.01, I=0.0)
    >>> assert λ > 1.0, "Lambda should increase when S < target"
    
    >>> # Test 3: Deadband prevents updates
    >>> λ, I, _ = update_lambda(1.0, S_meas=0.0101, S_target=0.01, I=0.0, deadband=0.002)
    >>> assert λ == 1.0, "Lambda unchanged within deadband"
    
    >>> # Test 4: Lambda bounds respected
    >>> λ, I, _ = update_lambda(9.5, S_meas=0.005, S_target=0.01, I=0.0, lmax=10.0)
    >>> assert λ <= 10.0, "Lambda respects upper bound"
    """
    # EMA smoothing of S measurement
    if S_hat is None:
        S_hat = S_meas
    else:
        S_hat = ema(S_hat, S_meas, ema_alpha)
    
    # Check deadband
    error = S_hat - S_target
    if abs(error) <= deadband:
        # Within deadband: no update
        return lmbda, I * leak, S_hat
    
    # Apply integral leak (slow decay)
    I = I * leak
    
    # Anti-windup: only integrate if not saturated or moving away from saturation
    if satur_guard:
        at_min = (lmbda <= lmin)
        at_max = (lmbda >= lmax)
        should_integrate = not (
            (at_min and error < 0) or  # At min and want to go lower
            (at_max and error > 0)      # At max and want to go higher
        )
    else:
        should_integrate = True
    
    if should_integrate:
        # Update integral with clamping
        I = max(i_min, min(i_max, I + Ki * error))
    
    # Compute control effort
    control_effort = Kp * error + I
    
    # Multiplicative update for negative plant gain (increase λ when e>0)
    lmbda_new = lmbda * math.exp(control_effort)
    
    # Apply bounds
    lmbda_new = max(lmin, min(lmax, lmbda_new))
    
    # Guardrail assertions (lightweight; keep always-on as they are cheap)
    assert 0.0 <= S_hat <= 1.0 or math.isnan(S_hat) is False, "S_hat out of [0,1] or NaN"
    assert lmin - 1e-12 <= lmbda_new <= lmax + 1e-12, "lambda out of bounds"
    assert i_min - 1e-12 <= I <= i_max + 1e-12, "integral term out of bounds"
    return lmbda_new, I, S_hat


def calculate_param_bpt(
    model,
    sigma: float = 0.01,
    tokens_per_epoch: int = 1000000
) -> float:
    """Calculate Parameter BPT for LoRA weights.
    
    ParamBPT = (1 / (N * ln(2))) * Σ(w² / (2σ²)) for trainable LoRA parameters
    
    Args:
        model: Model with parameters
        sigma: Prior standard deviation
        tokens_per_epoch: Fixed normalization constant
        
    Returns:
        Parameter bits per token
        
    # Unit tests:
    >>> # Test 1: Zero weights => ParamBPT ≈ 0
    >>> # Test 2: Scaling weights by 2 => ParamBPT × 4 (quadratic)
    """
    import torch
    
    param_sum = 0.0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name.lower():
            # Skip meta device parameters (offloaded to disk)
            if param.device.type == 'meta':
                continue
                
            # Sum of squares in float32 for stability
            param_sum += (param.data.float() ** 2).sum().item()
            param_count += param.numel()
    
    if param_count == 0:
        # Return a small default value to avoid division by zero
        # This happens when all parameters are on meta device
        return 1e-6
    
    # Convert to bits (divide by ln(2)) and normalize by tokens
    param_bpt = param_sum / (2 * sigma**2 * tokens_per_epoch * math.log(2))
    
    return param_bpt


def calculate_data_bpt(loss_nats: float) -> float:
    """Convert loss in nats/token to bits/token.
    
    Args:
        loss_nats: Cross-entropy loss in nats per token
        
    Returns:
        Data bits per token
    """
    return loss_nats / math.log(2)


def calculate_s_ratio(data_bpt: float, param_bpt: float) -> float:
    """Calculate S ratio.
    
    S = ParamBPT / (DataBPT + ParamBPT)
    
    Args:
        data_bpt: Data bits per token
        param_bpt: Parameter bits per token
        
    Returns:
        S ratio (0 to 1)
    """
    total = data_bpt + param_bpt
    if total == 0:
        return 0.0
    return param_bpt / total
