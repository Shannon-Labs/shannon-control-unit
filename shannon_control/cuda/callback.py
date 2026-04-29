"""
SCU Training Callback for PyTorch / HuggingFace Trainer

Integrates Shannon Control Unit's PI controller into the PyTorch training loop.
Monitors the S-ratio and adjusts regularization strength (lambda) using PI control.

Works with both raw PyTorch loops and HuggingFace Trainer via TrainerCallback.
"""

import math
import time
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from ..control import (
    update_lambda,
    calculate_data_bpt,
    calculate_s_ratio,
    calculate_param_bpt,
    calculate_param_bpt_from_stats,
)


@dataclass
class SCUState:
    """Persistent state for SCU PI controller across training."""

    lambda_: float = 1.0
    I: float = 0.0
    S_hat: Optional[float] = None
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
    gpu_memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    timestamp: float = field(default_factory=time.time)


class SCUCUDACallback:
    """
    PyTorch SCU callback implementing PI control for CUDA training.

    Provides:
    - Real-time S-ratio monitoring
    - Adaptive lambda (regularization) control via PI controller
    - GPU memory and utilization monitoring
    - Comprehensive metrics logging to CSV/JSON
    - MDL saturation detection (self-terminating signal)

    Usage with raw training loop:
        callback = SCUCUDACallback(model=model, tokens_per_epoch=1000000, ...)
        for step, batch in enumerate(dataloader):
            loss = model(**batch).loss
            callback.on_step(step, loss.item())
            reg_loss = callback.compute_regularization_loss()
            total_loss = loss + reg_loss
            total_loss.backward()

    Usage with HuggingFace Trainer:
        callback = SCUCUDACallback(model=model, ...)
        # Wrap as TrainerCallback — see SCUTrainerCallback below
    """

    def __init__(
        self,
        model,
        tokens_per_epoch: int,
        target_s: float = 0.04,
        control_frequency: int = 50,
        Kp: float = 0.9,
        Ki: float = 0.18,
        deadband: float = 0.003,
        lambda_init: float = 1.0,
        lambda_min: float = 1e-4,
        lambda_max: float = 2.0,
        prior_sigma: float = 0.01,
        log_dir: Optional[str] = None,
        log_csv: Optional[str] = None,
    ):
        self.model = model
        self.target_s = target_s
        self.control_frequency = control_frequency
        self.Kp = Kp
        self.Ki = Ki
        self.deadband = deadband
        self.prior_sigma = prior_sigma
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Controller state
        self.state = SCUState(
            lambda_=lambda_init,
            tokens_per_epoch=tokens_per_epoch,
        )

        # Metrics history
        self.metrics_history: List[SCUMetrics] = []
        self.last_control_step = 0

        # MDL saturation detection
        self._lambda_history: List[float] = []
        self._saturation_window = 100
        self._saturation_threshold = 0.001

        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.csv_file = None
        self.csv_writer = None
        if log_csv:
            csv_path = Path(log_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = open(csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "step", "loss_nats", "data_bpt", "param_bpt", "s_ratio",
                "lambda", "integral", "gpu_mem_gb", "gpu_util", "timestamp",
            ])

    def on_step(self, step: int, loss_nats: float):
        """
        Called after each training step.

        Computes S-ratio, applies PI control, and logs metrics.

        Args:
            step: Current training step
            loss_nats: Cross-entropy loss in nats
        """
        # Compute BPT metrics
        data_bpt = calculate_data_bpt(loss_nats)
        param_bpt = calculate_param_bpt(
            self.model,
            tokens_per_epoch=self.state.tokens_per_epoch,
            sigma=self.prior_sigma,
        )
        s_ratio = calculate_s_ratio(data_bpt, param_bpt)

        # GPU metrics
        gpu_mem_gb, gpu_util = self._get_gpu_metrics()

        # Apply PI control at specified frequency
        if step - self.last_control_step >= self.control_frequency:
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
            self.last_control_step = step

            # Track lambda for saturation detection
            self._lambda_history.append(new_lambda)

        # Record metrics
        metrics = SCUMetrics(
            step=step,
            loss_nats=loss_nats,
            data_bpt=data_bpt,
            param_bpt=param_bpt,
            s_ratio=s_ratio,
            lambda_value=self.state.lambda_,
            integral_term=self.state.I,
            gpu_memory_gb=gpu_mem_gb,
            gpu_utilization=gpu_util,
        )
        self.metrics_history.append(metrics)

        # CSV logging
        if self.csv_writer:
            self.csv_writer.writerow([
                step,
                f"{loss_nats:.6f}",
                f"{data_bpt:.6f}",
                f"{param_bpt:.8f}",
                f"{s_ratio:.6f}",
                f"{self.state.lambda_:.6f}",
                f"{self.state.I:.6f}",
                f"{gpu_mem_gb:.2f}",
                f"{gpu_util:.1f}",
                f"{time.time():.2f}",
            ])
            self.csv_file.flush()

    def compute_regularization_loss(self) -> "torch.Tensor":
        """
        Compute the SCU regularization term: lambda * sum(w^2) / (2 * sigma^2).

        Returns a differentiable torch scalar to add to the CE loss.
        """
        import torch

        reg = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                if param.device.type != "meta":
                    reg = reg + (param.float() ** 2).sum()

        reg = self.state.lambda_ * reg / (2.0 * self.prior_sigma ** 2)

        # Normalize by tokens_per_epoch to match BPT scale
        reg = reg / (self.state.tokens_per_epoch * math.log(2))

        return reg

    def is_saturated(self) -> bool:
        """
        Check if lambda has converged (MDL saturation signal).

        Returns True when lambda has been stable for _saturation_window
        control steps, indicating the model has learned all it can.
        """
        if len(self._lambda_history) < self._saturation_window:
            return False

        recent = self._lambda_history[-self._saturation_window:]
        delta = max(recent) - min(recent)
        return delta < self._saturation_threshold

    def get_current_lambda(self) -> float:
        """Return current regularization strength."""
        return self.state.lambda_

    def _get_gpu_metrics(self):
        """Get GPU memory usage and utilization."""
        try:
            import torch
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                # pynvml for utilization if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    return mem_gb, util.gpu
                except Exception:
                    return mem_gb, 0.0
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0

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
            "lambda_range": [min(lambdas), max(lambdas)] if lambdas else [0.0, 0.0],
            "final_loss": losses[-1] if losses else 0.0,
            "mean_loss": sum(losses) / len(losses) if losses else 0.0,
            "target_s": self.target_s,
            "saturated": self.is_saturated(),
        }

    def save_metrics(self, output_path: Optional[str] = None):
        """Save training metrics to JSON file."""
        path = Path(output_path) if output_path else (
            self.log_dir / "scu_metrics.json" if self.log_dir else None
        )
        if not path:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = [
            {
                "step": m.step,
                "loss_nats": m.loss_nats,
                "data_bpt": m.data_bpt,
                "param_bpt": m.param_bpt,
                "s_ratio": m.s_ratio,
                "lambda": m.lambda_value,
                "integral": m.integral_term,
                "gpu_mem_gb": m.gpu_memory_gb,
                "gpu_util": m.gpu_utilization,
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
            },
            "summary": self.get_metrics_summary(),
            "metrics": metrics_data,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[SCU] Metrics saved to {path}")

    def shutdown(self):
        """Clean shutdown."""
        if self.csv_file:
            self.csv_file.close()
        if self.log_dir and self.metrics_history:
            self.save_metrics()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
