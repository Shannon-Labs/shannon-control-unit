"""
Simplified Shannon Control Unit (S-SCU)

Focuses on computational efficiency and training quality rather than unnecessary
thermodynamic simulation. Keeps the useful parts like multi-scale entropy analysis
for understanding training dynamics.
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingState:
    """Current training state for optimization"""
    loss_value: float
    gradient_norm: float
    learning_rate: float
    batch_size: int
    step_count: int
    timestamp: float


@dataclass
class ControlAction:
    """Control action for training optimization"""
    learning_rate_factor: float
    batch_size_adjustment: int
    gradient_clipping_factor: float
    regularization_factor: float
    reason: str


class SimplifiedSCU:
    """
    Simplified Shannon Control Unit focused on computational efficiency.

    Uses multi-scale entropy analysis to understand training dynamics and
    makes simple, practical adjustments to improve training efficiency.
    """

    def __init__(
        self,
        target_loss_improvement: float = 0.01,  # 1% improvement per 100 steps
        control_frequency: int = 100,  # Control every 100 steps
        enable_multiscale_analysis: bool = True
    ):
        """
        Initialize Simplified SCU.

        Args:
            target_loss_improvement: Target loss improvement rate
            control_frequency: How often to apply control (in steps)
            enable_multiscale_analysis: Enable multi-scale entropy analysis
        """
        self.target_loss_improvement = target_loss_improvement
        self.control_frequency = control_frequency
        self.enable_multiscale_analysis = enable_multiscale_analysis

        # Training history
        self.loss_history: List[float] = []
        self.gradient_history: List[float] = []
        self.lr_history: List[float] = []
        self.max_history_length = 1000

        # Multi-scale analyzer if enabled
        if self.enable_multiscale_analysis:
            try:
                from .multiscale_entropy import MultiScaleEntropyAnalyzer
                self.multiscale_analyzer = MultiScaleEntropyAnalyzer(max_scales=6)
            except ImportError:
                print("Multi-scale entropy analysis not available, disabling...")
                self.multiscale_analyzer = None
                self.enable_multiscale_analysis = False
        else:
            self.multiscale_analyzer = None

        # Control statistics
        self.control_actions_applied = 0
        self.last_control_step = 0

    def update_state(self, state: TrainingState) -> Optional[ControlAction]:
        """
        Update training state and recommend control actions if needed.

        Args:
            state: Current training state

        Returns:
            Control action if needed, None otherwise
        """
        # Update history
        self.loss_history.append(state.loss_value)
        self.gradient_history.append(state.gradient_norm)
        self.lr_history.append(state.learning_rate)

        # Trim history
        if len(self.loss_history) > self.max_history_length:
            self.loss_history = self.loss_history[-self.max_history_length:]
            self.gradient_history = self.gradient_history[-self.max_history_length:]
            self.lr_history = self.lr_history[-self.max_history_length:]

        # Check if we should apply control
        if (state.step_count - self.last_control_step) < self.control_frequency:
            return None

        return self._analyze_and_recommend(state)

    def _analyze_and_recommend(self, state: TrainingState) -> Optional[ControlAction]:
        """Analyze training state and recommend control actions"""
        if len(self.loss_history) < 50:  # Need enough history
            return None

        # Calculate recent trends
        recent_loss = np.mean(self.loss_history[-10:])
        earlier_loss = np.mean(self.loss_history[-50:-10])

        if earlier_loss == 0:
            return None

        loss_improvement = (earlier_loss - recent_loss) / earlier_loss

        # Analyze gradient stability
        recent_grads = np.array(self.gradient_history[-20:])
        grad_std = np.std(recent_grads)
        grad_mean = np.mean(recent_grads)
        grad_cv = grad_std / (grad_mean + 1e-8)  # Coefficient of variation

        # Multi-scale analysis if available
        multiscale_insights = {}
        if self.enable_multiscale_analysis and self.multiscale_analyzer:
            try:
                multiscale_result = self.multiscale_analyzer.analyze_training_signal(
                    self.loss_history[-64:],
                    self.gradient_history[-64:],
                    temperature=300.0  # Dummy temp, not actually used
                )
                multiscale_insights = {
                    'entropy_complexity': len(multiscale_result.selected_scales),
                    'landauer_residual': multiscale_result.landauer_residual_bits,
                    'total_entropy': multiscale_result.total_entropy_bits
                }
            except:
                pass

        # Make recommendations based on analysis
        action = self._make_control_decision(
            loss_improvement, grad_cv, multiscale_insights, state
        )

        if action:
            self.control_actions_applied += 1
            self.last_control_step = state.step_count

        return action

    def _make_control_decision(
        self,
        loss_improvement: float,
        grad_cv: float,
        multiscale_insights: Dict,
        state: TrainingState
    ) -> Optional[ControlAction]:
        """Make control decision based on analysis"""

        # Check if loss is improving too slowly
        if loss_improvement < self.target_loss_improvement:
            if grad_cv > 2.0:  # Unstable gradients
                return ControlAction(
                    learning_rate_factor=0.8,
                    batch_size_adjustment=0,
                    gradient_clipping_factor=0.9,
                    regularization_factor=1.2,
                    reason="Slow loss improvement with unstable gradients - reducing LR and increasing regularization"
                )
            else:
                return ControlAction(
                    learning_rate_factor=1.2,
                    batch_size_adjustment=0,
                    gradient_clipping_factor=1.0,
                    regularization_factor=1.0,
                    reason="Slow loss improvement with stable gradients - increasing LR"
                )

        # Check for overfitting or convergence issues
        if loss_improvement < 0:  # Loss getting worse
            if multiscale_insights.get('entropy_complexity', 0) > 4:
                return ControlAction(
                    learning_rate_factor=0.7,
                    batch_size_adjustment=2,
                    gradient_clipping_factor=0.8,
                    regularization_factor=1.5,
                    reason="Loss degradation with high entropy complexity - strong regularization needed"
                )
            else:
                return ControlAction(
                    learning_rate_factor=0.9,
                    batch_size_adjustment=0,
                    gradient_clipping_factor=0.9,
                    regularization_factor=1.1,
                    reason="Loss degradation - mild regularization"
                )

        # Check if training is too stable (might be stuck in local minimum)
        if abs(loss_improvement) < 0.001 and grad_cv < 0.5:
            return ControlAction(
                learning_rate_factor=1.1,
                batch_size_adjustment=1,
                gradient_clipping_factor=1.1,
                regularization_factor=0.9,
                reason="Training too stable - encouraging exploration"
            )

        return None  # No action needed

    def get_training_summary(self) -> Dict:
        """Get summary of training and control actions"""
        if not self.loss_history:
            return {"status": "no_data"}

        return {
            'total_steps': len(self.loss_history),
            'current_loss': self.loss_history[-1],
            'loss_trend': (self.loss_history[-1] - self.loss_history[-10]) / 10 if len(self.loss_history) > 10 else 0,
            'control_actions_applied': self.control_actions_applied,
            'average_gradient_norm': np.mean(self.gradient_history[-100:]) if self.gradient_history else 0,
            'current_learning_rate': self.lr_history[-1] if self.lr_history else 0,
            'multiscale_enabled': self.enable_multiscale_analysis
        }


# Convenience function for quick setup
def create_simplified_scu(**kwargs) -> SimplifiedSCU:
    """Create a simplified SCU with default settings"""
    return SimplifiedSCU(**kwargs)


if __name__ == "__main__":
    # Test the simplified controller
    print("Testing Simplified SCU...")

    controller = create_simplified_scu(control_frequency=10)

    # Simulate training
    for step in range(100):
        # Simulate loss decreasing with some noise
        base_loss = 2.0 * math.exp(-step * 0.01)
        noise = 0.1 * np.random.randn()
        loss = base_loss + noise

        # Simulate gradient norm
        grad_norm = 0.5 + 0.3 * math.sin(step * 0.1) + 0.1 * np.random.randn()

        state = TrainingState(
            loss_value=loss,
            gradient_norm=grad_norm,
            learning_rate=2e-4,
            batch_size=2,
            step_count=step,
            timestamp=time.time()
        )

        action = controller.update_state(state)
        if action:
            print(f"Step {step}: {action.reason}")

    summary = controller.get_training_summary()
    print(f"\nTraining Summary: {summary}")
    print("Simplified SCU test completed!")