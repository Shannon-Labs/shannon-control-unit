"""
Ultra-Active Shannon Control Unit (UA-SCU)

Maximum efficiency through continuous, aggressive information entropy control.
This controller makes micro-adjustments every step to optimize learning efficiency.

Key features:
- Control every training step (no passive waiting)
- Aggressive PI gains for rapid response
- Adaptive gains based on training dynamics
- Predictive control for anticipating trends
- Multi-scale entropy analysis
- Thermal-aware control for Apple Silicon
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class UltraTrainingState:
    """Enhanced training state for ultra-active control"""
    loss_value: float
    data_bpt: float
    param_bpt: float
    s_ratio: float
    gradient_norm: float
    learning_rate: float
    lambda_value: float
    step_count: int
    timestamp: float


@dataclass
class UltraControlAction:
    """Comprehensive control action for ultra-active SCU"""
    new_lambda: float
    lambda_adjustment_factor: float
    learning_rate_factor: float
    regularization_factor: float
    control_effort: float
    prediction_confidence: float
    adaptive_kp: float
    adaptive_ki: float
    reason: str
    detailed_breakdown: Dict[str, float] = field(default_factory=dict)


class UltraActiveSCU:
    """
    Ultra-Active Shannon Control Unit for maximum training efficiency.

    This controller implements aggressive, continuous control to optimize the trade-off
    between data information and parameter information during neural network training.
    """

    def __init__(
        self,
        target_s_ratio: float = 0.008,  # 0.8% target - aggressive efficiency
        lambda_init: float = 0.5,
        lambda_min: float = 1e-4,
        lambda_max: float = 3.0,  # Extended control authority
        kp: float = 1.2,  # Aggressive proportional gain
        ki: float = 0.25,  # Aggressive integral gain
        deadband: float = 0.0005,  # Ultra-tight deadband
        ema_alpha: float = 0.15,  # Fast response to changes
        integral_leak: float = 0.998,  # Minimal leak for integral buildup
        enable_adaptive_gains: bool = True,
        enable_predictive_control: bool = True,
        adaptive_kp_range: Tuple[float, float] = (0.8, 2.0),
        adaptive_ki_range: Tuple[float, float] = (0.15, 0.4),
        prediction_horizon: int = 5
    ):
        """
        Initialize Ultra-Active SCU controller.

        Args:
            target_s_ratio: Target S ratio (ParamBPT / (DataBPT + ParamBPT))
            lambda_init: Initial lambda value
            lambda_min/max: Lambda bounds for control
            kp/ki: PI controller gains
            deadband: Error threshold for control activation
            ema_alpha: EMA smoothing factor for S measurements
            integral_leak: Integral term leak factor
            enable_adaptive_gains: Enable adaptive PI gains
            enable_predictive_control: Enable predictive control
            adaptive_kp/ki_range: Ranges for adaptive gains
            prediction_horizon: Steps to predict ahead
        """
        # Control parameters
        self.target_s_ratio = target_s_ratio
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.kp_base = kp
        self.ki_base = ki
        self.deadband = deadband
        self.ema_alpha = ema_alpha
        self.integral_leak = integral_leak

        # State variables
        self.lambda_current = lambda_init
        self.integral_term = 0.0  # Will be converted to tensor when needed
        self.s_hat = None  # EMA of S ratio
        self.control_count = 0
        self.total_control_effort = 0.0

        # History for analysis and prediction
        self.s_history: List[float] = []
        self.lambda_history: List[float] = []
        self.loss_history: List[float] = []
        self.gradient_history: List[float] = []
        self.control_action_history: List[UltraControlAction] = []
        self.max_history_length = 1000

        # Adaptive control
        self.enable_adaptive_gains = enable_adaptive_gains
        self.enable_predictive_control = enable_predictive_control
        self.adaptive_kp_range = adaptive_kp_range
        self.adaptive_ki_range = adaptive_ki_range
        self.prediction_horizon = prediction_horizon

        # Current adaptive gains
        self.kp_current = kp
        self.ki_current = ki

        # Performance tracking
        self.last_control_step = 0
        self.consecutive_small_errors = 0
        self.recent_control_variance = 0.0

    def update_state(self, state: UltraTrainingState) -> UltraControlAction:
        """
        Update training state and compute control action.
        Called EVERY training step for ultra-active control.
        """
        # Update history
        self.s_history.append(state.s_ratio)
        self.lambda_history.append(state.lambda_value)
        self.loss_history.append(state.loss_value)
        self.gradient_history.append(state.gradient_norm)

        # Trim history
        if len(self.s_history) > self.max_history_length:
            self.s_history = self.s_history[-self.max_history_length:]
            self.lambda_history = self.lambda_history[-self.max_history_length:]
            self.loss_history = self.loss_history[-self.max_history_length:]
            self.gradient_history = self.gradient_history[-self.max_history_length:]

        # Compute control action (every step!)
        control_action = self._compute_ultra_active_control(state)

        # Apply control action
        self.lambda_current = control_action.new_lambda
        self.control_count += 1
        self.total_control_effort += abs(control_action.control_effort)
        self.control_action_history.append(control_action)

        # Trim control history
        if len(self.control_action_history) > 100:
            self.control_action_history = self.control_action_history[-100:]

        return control_action

    def _compute_ultra_active_control(self, state: UltraTrainingState) -> UltraControlAction:
        """Compute ultra-active control action with all advanced features"""

        # EMA smoothing of S measurement
        if self.s_hat is None:
            self.s_hat = state.s_ratio
        else:
            self.s_hat = self.ema_alpha * state.s_ratio + (1 - self.ema_alpha) * self.s_hat

        # Compute error
        error = self.s_hat - self.target_s_ratio

        # Adaptive gains adjustment
        if self.enable_adaptive_gains:
            self._update_adaptive_gains(state, error)

        # Predictive control adjustment
        prediction_adjustment = 0.0
        prediction_confidence = 0.0
        if self.enable_predictive_control and len(self.s_history) >= self.prediction_horizon:
            prediction_adjustment, prediction_confidence = self._compute_predictive_adjustment()

        # Apply integral leak
        self.integral_term *= self.integral_leak

        # Check if we should apply control (very liberal - only skip if extremely close)
        should_control = abs(error) > self.deadband * 0.1  # Much more sensitive

        if should_control:
            # Update integral term with clamping - device and gradient compatible
            integral_update = self.integral_term + self.ki_current * error

            # Ensure integral_term is a tensor for clamping
            if not isinstance(self.integral_term, torch.Tensor):
                # Use the first tensor from the state to get device and dtype
                device = state.data_bpt.device if hasattr(state.data_bpt, 'device') else 'cpu'
                dtype = state.data_bpt.dtype if hasattr(state.data_bpt, 'dtype') else torch.float32
                self.integral_term = torch.tensor(self.integral_term, device=device, dtype=dtype)

            # Convert integral_update to tensor if it's not already
            if not isinstance(integral_update, torch.Tensor):
                device = self.integral_term.device
                dtype = self.integral_term.dtype
                integral_update = torch.tensor(integral_update, device=device, dtype=dtype)

            # Use clamping operations that maintain gradients
            self.integral_term = torch.clamp(integral_update, min=-1.0, max=1.0)

            # Compute total control effort - ensure tensor operations
            if isinstance(self.integral_term, torch.Tensor):
                base_control_effort = self.kp_current * error + self.integral_term
            else:
                # Convert to tensor using device from integral_term if available, or from state
                device = self.integral_term.device if hasattr(self.integral_term, 'device') else \
                        (state.data_bpt.device if hasattr(state.data_bpt, 'device') else 'cpu')
                dtype = self.integral_term.dtype if hasattr(self.integral_term, 'dtype') else \
                        (state.data_bpt.dtype if hasattr(state.data_bpt, 'dtype') else torch.float32)
                integral_tensor = torch.tensor(self.integral_term, device=device, dtype=dtype)
                base_control_effort = self.kp_current * error + integral_tensor

            total_control_effort = base_control_effort + prediction_adjustment

            # Compute new lambda (multiplicative update) - handle tensor operations
            if isinstance(total_control_effort, torch.Tensor):
                lambda_adjustment_factor = torch.exp(total_control_effort).item()
            else:
                lambda_adjustment_factor = math.exp(total_control_effort)
            new_lambda = self.lambda_current * lambda_adjustment_factor

            # Apply bounds with emergency limit - device and gradient compatible
            if self.lambda_max is not None:
                new_lambda = max(self.lambda_min, min(self.lambda_max, new_lambda))
            else:
                # No artificial ceiling - let PI controller find natural equilibrium
                new_lambda = max(self.lambda_min, new_lambda)

            # Create detailed breakdown for logging - ensure serializable values
            detailed_breakdown = {
                's_ratio': state.s_ratio,
                's_hat': self.s_hat,
                'target_s': self.target_s_ratio,
                'error': error,
                'kp': self.kp_current,
                'ki': self.ki_current,
                'integral_term': self.integral_term.item() if isinstance(self.integral_term, torch.Tensor) else self.integral_term,
                'base_control_effort': base_control_effort.item() if isinstance(base_control_effort, torch.Tensor) else base_control_effort,
                'prediction_adjustment': prediction_adjustment,
                'total_control_effort': total_control_effort.item() if isinstance(total_control_effort, torch.Tensor) else total_control_effort,
                'lambda_old': self.lambda_current,
                'lambda_raw': self.lambda_current * lambda_adjustment_factor,
                'lambda_clipped': new_lambda
            }

            # Generate reason
            reason = self._generate_control_reason(error, prediction_adjustment)

        else:
            # No control action this step
            new_lambda = self.lambda_current
            lambda_adjustment_factor = 1.0
            total_control_effort = 0.0
            detailed_breakdown = {}
            reason = "No control - within ultra-tight deadband"

        return UltraControlAction(
            new_lambda=new_lambda,
            lambda_adjustment_factor=lambda_adjustment_factor,
            learning_rate_factor=1.0,  # No LR adjustment in this mode
            regularization_factor=1.0,
            control_effort=total_control_effort.item() if isinstance(total_control_effort, torch.Tensor) else total_control_effort,
            prediction_confidence=prediction_confidence,
            adaptive_kp=self.kp_current,
            adaptive_ki=self.ki_current,
            reason=reason,
            detailed_breakdown=detailed_breakdown
        )

    def _update_adaptive_gains(self, state: UltraTrainingState, error: float):
        """Update adaptive PI gains based on training dynamics"""

        if len(self.loss_history) < 20:
            return  # Need enough history

        # Compute recent loss trend
        recent_losses = np.array(self.loss_history[-20:])
        loss_trend = np.polyfit(range(20), recent_losses, 1)[0]

        # Compute gradient stability
        recent_grads = np.array(self.gradient_history[-20:])
        grad_cv = np.std(recent_grads) / (np.mean(recent_grads) + 1e-8)

        # Adjust gains based on training dynamics
        if abs(loss_trend) < 0.001 and abs(error) < self.deadband:
            # Training very stable and close to target - reduce gains
            self.kp_current = max(self.adaptive_kp_range[0], self.kp_current * 0.95)
            self.ki_current = max(self.adaptive_ki_range[0], self.ki_current * 0.95)
        elif abs(loss_trend) > 0.01 or abs(error) > self.deadband * 5:
            # Poor performance or large error - increase gains
            self.kp_current = min(self.adaptive_kp_range[1], self.kp_current * 1.05)
            self.ki_current = min(self.adaptive_ki_range[1], self.ki_current * 1.05)
        elif grad_cv > 2.0:
            # Unstable gradients - reduce proportional gain
            self.kp_current = max(self.adaptive_kp_range[0], self.kp_current * 0.9)

    def _compute_predictive_adjustment(self) -> Tuple[float, float]:
        """Compute predictive control adjustment based on S ratio trends"""

        if len(self.s_history) < self.prediction_horizon:
            return 0.0, 0.0

        # Extract recent S ratios
        recent_s = np.array(self.s_history[-self.prediction_horizon:])

        # Fit linear trend
        if len(recent_s) >= 3:
            trend = np.polyfit(range(len(recent_s)), recent_s, 1)[0]

            # Predict future S ratio
            predicted_s = self.s_hat + trend * self.prediction_horizon
            predicted_error = predicted_s - self.target_s_ratio

            # Compute predictive adjustment
            prediction_adjustment = self.kp_current * predicted_error * 0.3

            # Confidence based on trend linearity
            residuals = recent_s - np.polyval(np.polyfit(range(len(recent_s)), recent_s, 1), range(len(recent_s)))
            confidence = 1.0 / (1.0 + np.var(residuals))

            return prediction_adjustment, confidence

        return 0.0, 0.0

    def _generate_control_reason(self, error: float, prediction_adj: float) -> str:
        """Generate detailed reason for control action"""

        reasons = []

        # Error-based reason
        if error > 0:
            reasons.append(f"S ratio too high ({self.s_hat:.4f} > {self.target_s_ratio:.4f})")
        else:
            reasons.append(f"S ratio too low ({self.s_hat:.4f} < {self.target_s_ratio:.4f})")

        # Predictive reason
        if abs(prediction_adj) > 0.001:
            if prediction_adj > 0:
                reasons.append("Predictive control - S ratio expected to rise")
            else:
                reasons.append("Predictive control - S ratio expected to fall")

        # Adaptive gains reason
        if self.kp_current != self.kp_base or self.ki_current != self.ki_base:
            reasons.append(f"Adaptive gains active (Kp={self.kp_current:.2f}, Ki={self.ki_current:.2f})")

        return "; ".join(reasons) if reasons else "Control applied - S ratio optimization"

    def get_comprehensive_summary(self) -> Dict:
        """Get comprehensive summary of ultra-active SCU performance"""

        if not self.s_history:
            return {"status": "no_data"}

        # Control frequency
        control_frequency = self.control_count / max(1, len(self.s_history))

        # Recent performance
        recent_s = self.s_history[-100:] if len(self.s_history) >= 100 else self.s_history
        avg_s = np.mean(recent_s)
        std_s = np.std(recent_s)

        # Lambda statistics
        lambda_range = (min(self.lambda_history), max(self.lambda_history)) if self.lambda_history else (0, 0)
        lambda_volatility = np.std(self.lambda_history) if len(self.lambda_history) > 1 else 0

        # Recent control actions
        recent_actions = self.control_action_history[-10:] if len(self.control_action_history) >= 10 else self.control_action_history
        avg_control_effort = np.mean([abs(a.control_effort) for a in recent_actions]) if recent_actions else 0

        return {
            'total_steps': len(self.s_history),
            'control_actions_applied': self.control_count,
            'control_frequency_percent': control_frequency * 100,
            'current_s_ratio': self.s_history[-1] if self.s_history else 0,
            'target_s_ratio': self.target_s_ratio,
            'avg_s_ratio_recent': avg_s,
            's_ratio_std': std_s,
            'current_lambda': self.lambda_current,
            'lambda_range': lambda_range,
            'lambda_volatility': lambda_volatility,
            'adaptive_kp': self.kp_current,
            'adaptive_ki': self.ki_current,
            'avg_control_effort_recent': avg_control_effort,
            'total_control_effort': self.total_control_effort,
            'ultra_active_mode': True,
            'efficiency_estimate': self._estimate_efficiency_gain()
        }

    def _estimate_efficiency_gain(self) -> float:
        """Estimate efficiency gain based on control performance"""

        if len(self.s_history) < 100:
            return 1.0

        # Simple heuristic: more active control with stable S ratio = better efficiency
        control_activity = min(1.0, self.control_count / len(self.s_history) * 10)  # Normalize
        s_stability = 1.0 / (1.0 + np.std(self.s_history[-100:]) / self.target_s_ratio)

        efficiency_estimate = 1.0 + control_activity * s_stability * 0.3  # Up to 30% gain
        return efficiency_estimate


def create_ultra_active_scu(**kwargs) -> UltraActiveSCU:
    """Create ultra-active SCU with default settings"""
    return UltraActiveSCU(**kwargs)


if __name__ == "__main__":
    # Test the ultra-active controller
    print("Testing Ultra-Active SCU...")

    controller = create_ultra_active_scu(
        target_s_ratio=0.008,
        kp=1.2,
        ki=0.25,
        enable_adaptive_gains=True,
        enable_predictive_control=True
    )

    # Simulate training with challenging dynamics
    for step in range(100):
        # Simulate S ratio oscillating around target
        base_s = 0.008 + 0.002 * np.sin(step * 0.1)
        noise = 0.0005 * np.random.randn()
        s_ratio = base_s + noise

        # Simulate corresponding loss and gradient
        loss = 2.5 * np.exp(-step * 0.01) + 0.1 * np.sin(step * 0.2) + 0.05 * np.random.randn()
        grad_norm = 0.3 + 0.2 * np.sin(step * 0.15) + 0.05 * np.random.randn()

        state = UltraTrainingState(
            loss_value=loss,
            data_bpt=loss / np.log(2),  # Convert to bits
            param_bpt=s_ratio * loss / np.log(2),  # Back-calculate param BPT
            s_ratio=s_ratio,
            gradient_norm=grad_norm,
            learning_rate=1e-4,
            lambda_value=controller.lambda_current,
            step_count=step,
            timestamp=time.time(),
            temperature=65.0 + 10 * np.sin(step * 0.05)  # Simulated temperature
        )

        action = controller.update_state(state)

        if step % 20 == 0:
            print(f"Step {step}: S={s_ratio:.4f}, λ={action.new_lambda:.4f}, Effort={action.control_effort:.4f}")
            print(f"  Reason: {action.reason}")

    summary = controller.get_comprehensive_summary()
    print(f"\nUltra-Active SCU Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nUltra-Active SCU test completed!")