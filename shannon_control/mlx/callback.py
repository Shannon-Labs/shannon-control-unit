"""
SCU Training Callback for mlx-lm

Implements the TrainingCallback interface from mlx-lm to integrate
Shannon Control Unit's PI controller into the native MLX training loop.

The callback monitors the S-ratio (information ratio) and automatically
adjusts the regularization strength (lambda) using PI control.
"""

import math
import time
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# Import SCU control functions (framework-agnostic)
from ..control import (
    update_lambda,
    calculate_data_bpt,
    calculate_s_ratio,
)
from ..mlx_adapters import calculate_param_bpt_mlx


@dataclass
class SCUState:
    """Persistent state for SCU PI controller across training."""

    lambda_: float = 1.0
    I: float = 0.0  # Integral term
    S_hat: Optional[float] = None  # EMA of S ratio
    tokens_per_epoch: int = 100000


@dataclass
class SCUMetrics:
    """Metrics recorded at each control step."""

    step: int
    loss_nats: float
    data_bpt: float
    param_bpt: float
    s_ratio: float
    lambda_value: float
    integral_term: float
    power_watts: float = 0.0
    temperature_celsius: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SCUTrainingCallback:
    """
    mlx-lm TrainingCallback implementing SCU PI control.

    Integrates with mlx-lm's train() function to provide:
    - Real-time S-ratio monitoring
    - Adaptive lambda (regularization) control
    - Power and thermal monitoring (optional)
    - Comprehensive metrics logging

    Usage:
        from mlx_lm.tuner.trainer import train
        from shannon_control.mlx import SCUTrainingCallback

        callback = SCUTrainingCallback(
            model=model,
            tokens_per_epoch=100000,
            target_s=0.01,
            log_dir="./scu_output"
        )

        train(model, tokenizer, ..., training_callback=callback)
    """

    def __init__(
        self,
        model,
        tokens_per_epoch: int,
        target_s: float = 0.01,
        control_frequency: int = 50,
        param_scope: str = "lora",
        param_bpt_frequency: Optional[int] = None,
        Kp: float = 0.8,
        Ki: float = 0.15,
        deadband: float = 0.002,
        lambda_init: float = 1.0,
        lambda_min: float = 1e-4,
        lambda_max: float = 2.0,
        prior_sigma: float = 0.01,
        enable_power_monitoring: bool = True,
        log_dir: Optional[str] = None,
        log_csv: Optional[str] = None,
    ):
        """
        Initialize SCU Training Callback.

        Args:
            model: MLX model reference (for param extraction)
            tokens_per_epoch: Normalization constant for BPT calculation
            target_s: Target S ratio (default 0.01 = 1%)
            control_frequency: Apply control every N steps (default 50)
            param_scope: "lora", "trainable", or "all"
            param_bpt_frequency: ParamBPT update interval (defaults to control frequency
                for non-LoRA scopes and every step for LoRA)
            Kp: Proportional gain for PI controller
            Ki: Integral gain for PI controller
            deadband: Error threshold below which no update occurs
            lambda_init: Initial regularization strength
            lambda_min: Minimum lambda bound
            lambda_max: Maximum lambda bound
            prior_sigma: Prior standard deviation for param BPT
            enable_power_monitoring: Enable Apple Silicon power monitoring
            log_dir: Directory for saving metrics and checkpoints
            log_csv: Path for CSV metrics log
        """
        self.model = model
        self.target_s = target_s
        self.control_frequency = control_frequency
        self.param_scope = param_scope
        if param_bpt_frequency is None:
            self.param_bpt_frequency = 1 if param_scope == "lora" else control_frequency
        else:
            self.param_bpt_frequency = param_bpt_frequency
        self.Kp = Kp
        self.Ki = Ki
        self.deadband = deadband
        self.prior_sigma = prior_sigma
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Controller state
        self.state = SCUState(
            lambda_=lambda_init,
            tokens_per_epoch=tokens_per_epoch
        )

        # Metrics history
        self.metrics_history: List[SCUMetrics] = []
        self.last_control_step = 0
        self.last_param_bpt_step = -1
        self.last_param_bpt: Optional[float] = None

        # Power monitoring
        self.power_monitor = None
        if enable_power_monitoring:
            self._init_power_monitor()

        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        if log_csv:
            csv_path = Path(log_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "step", "loss_nats", "data_bpt", "param_bpt", "s_ratio",
                "lambda", "integral", "power_watts", "temperature_c", "timestamp"
            ])

    def _init_power_monitor(self):
        """Initialize Apple Silicon power monitoring."""
        try:
            from ..hardware.apple_power_monitor import AppleSiliconPowerMonitor
            self.power_monitor = AppleSiliconPowerMonitor()
            if not self.power_monitor.initialize():
                print("[SCU] Warning: Power monitoring unavailable")
                self.power_monitor = None
        except ImportError:
            print("[SCU] Warning: Power monitor module not found")
            self.power_monitor = None
        except Exception as e:
            print(f"[SCU] Warning: Power monitor init failed: {e}")
            self.power_monitor = None

    def on_train_loss_report(self, train_info: dict):
        """
        Called by mlx-lm after each training step.

        This is the main integration point where we:
        1. Extract loss and compute S-ratio
        2. Apply PI control to adjust lambda
        3. Log metrics

        Args:
            train_info: Dict containing iteration, train_loss, learning_rate, etc.
        """
        iteration = train_info.get("iteration", 0)
        loss_nats = train_info.get("train_loss", 0.0)

        # Compute metrics
        data_bpt = calculate_data_bpt(loss_nats)
        if (
            self.last_param_bpt is None
            or iteration - self.last_param_bpt_step >= self.param_bpt_frequency
        ):
            self.last_param_bpt = calculate_param_bpt_mlx(
                self.model,
                self.state.tokens_per_epoch,
                self.prior_sigma,
                param_scope=self.param_scope,
            )
            self.last_param_bpt_step = iteration

        param_bpt = self.last_param_bpt if self.last_param_bpt is not None else 0.0
        s_ratio = calculate_s_ratio(data_bpt, param_bpt)

        # Get power metrics if available
        power_watts = 0.0
        temp_celsius = 0.0
        if self.power_monitor:
            readings = self.power_monitor.get_power_readings()
            if readings:
                power_watts = sum(r.power_watts for r in readings if r.power_watts)
                temps = [r.temperature_celsius for r in readings if r.temperature_celsius]
                temp_celsius = max(temps) if temps else 0.0

        # Apply PI control at specified frequency
        if iteration - self.last_control_step >= self.control_frequency:
            new_lambda, new_I, S_hat = update_lambda(
                lmbda=self.state.lambda_,
                S_meas=s_ratio,
                S_target=self.target_s,
                I=self.state.I,
                Kp=self.Kp,
                Ki=self.Ki,
                deadband=self.deadband,
                lmin=self.lambda_min,
                lmax=self.lambda_max,
                S_hat=self.state.S_hat,
            )

            self.state.lambda_ = new_lambda
            self.state.I = new_I
            self.state.S_hat = S_hat
            self.last_control_step = iteration

        # Record metrics
        metrics = SCUMetrics(
            step=iteration,
            loss_nats=loss_nats,
            data_bpt=data_bpt,
            param_bpt=param_bpt,
            s_ratio=s_ratio,
            lambda_value=self.state.lambda_,
            integral_term=self.state.I,
            power_watts=power_watts,
            temperature_celsius=temp_celsius,
        )
        self.metrics_history.append(metrics)

        # CSV logging
        if self.csv_writer:
            self.csv_writer.writerow([
                iteration,
                f"{loss_nats:.6f}",
                f"{data_bpt:.6f}",
                f"{param_bpt:.8f}",
                f"{s_ratio:.6f}",
                f"{self.state.lambda_:.6f}",
                f"{self.state.I:.6f}",
                f"{power_watts:.2f}",
                f"{temp_celsius:.1f}",
                f"{time.time():.2f}",
            ])
            self.csv_file.flush()

        # Periodic console logging
        if iteration % 100 == 0:
            print(
                f"[SCU] Step {iteration}: S={s_ratio:.4f} "
                f"(target={self.target_s:.4f}), lambda={self.state.lambda_:.4f}, "
                f"loss={loss_nats:.4f}"
            )

    def on_val_loss_report(self, val_info: dict):
        """
        Called by mlx-lm after validation.

        Args:
            val_info: Dict containing iteration, val_loss, etc.
        """
        iteration = val_info.get("iteration", 0)
        val_loss = val_info.get("val_loss", 0.0)

        print(
            f"[SCU] Validation at step {iteration}: val_loss={val_loss:.4f}, "
            f"current_lambda={self.state.lambda_:.4f}"
        )

    def get_current_lambda(self) -> float:
        """Return current regularization strength for external use."""
        return self.state.lambda_

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of training metrics."""
        if not self.metrics_history:
            return {}

        s_ratios = [m.s_ratio for m in self.metrics_history]
        lambdas = [m.lambda_value for m in self.metrics_history]
        losses = [m.loss_nats for m in self.metrics_history]

        return {
            "total_steps": len(self.metrics_history),
            "final_s_ratio": s_ratios[-1] if s_ratios else 0.0,
            "mean_s_ratio": sum(s_ratios) / len(s_ratios) if s_ratios else 0.0,
            "final_lambda": lambdas[-1] if lambdas else 0.0,
            "lambda_range": (min(lambdas), max(lambdas)) if lambdas else (0.0, 0.0),
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": sum(losses) / len(losses) if losses else 0.0,
            "target_s": self.target_s,
        }

    def save_metrics(self, output_path: Optional[str] = None):
        """Save training metrics to JSON file."""
        path = Path(output_path) if output_path else (
            self.log_dir / "scu_metrics.json" if self.log_dir else None
        )
        if not path:
            print("[SCU] Warning: No output path specified for metrics")
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert metrics to serializable format
        metrics_data = [
            {
                "step": m.step,
                "loss_nats": m.loss_nats,
                "data_bpt": m.data_bpt,
                "param_bpt": m.param_bpt,
                "s_ratio": m.s_ratio,
                "lambda": m.lambda_value,
                "integral": m.integral_term,
                "power_watts": m.power_watts,
                "temperature_c": m.temperature_celsius,
                "timestamp": m.timestamp,
            }
            for m in self.metrics_history
        ]

        output = {
            "config": {
                "target_s": self.target_s,
                "Kp": self.Kp,
                "Ki": self.Ki,
                "deadband": self.deadband,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
                "control_frequency": self.control_frequency,
                "param_scope": self.param_scope,
                "param_bpt_frequency": self.param_bpt_frequency,
            },
            "summary": self.get_metrics_summary(),
            "metrics": metrics_data,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[SCU] Metrics saved to {path}")

    def shutdown(self):
        """Clean shutdown of monitoring systems."""
        if self.power_monitor:
            self.power_monitor.shutdown()

        if self.csv_file:
            self.csv_file.close()

        # Auto-save metrics on shutdown
        if self.log_dir and self.metrics_history:
            self.save_metrics()

    def __del__(self):
        """Destructor to ensure clean shutdown."""
        try:
            self.shutdown()
        except Exception:
            pass
