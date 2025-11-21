"""
Power-Aware Training Integration for T-SCU

Integrates thermodynamic control into actual training loops, dynamically adjusting
training parameters based on real-time power consumption and efficiency metrics.

This is the main training interface that bridges T-SCU control with practical
deep learning training workflows.
"""

import time
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

# Import T-SCU components
from ..core.thermodynamic_controller import ThermodynamicSCU, ThermodynamicState
from ..hardware.power_monitor import MultiDevicePowerMonitor
from ..metrics.energy_entropy import EnergyEntropyMetrics


@dataclass
class PowerAwareTrainingConfig:
    """Configuration for power-aware training"""

    # Model configuration
    model_name: str = "Qwen/Qwen2.5-3B"
    use_lora: bool = True
    lora_rank: int = 16
    lora_dropout: float = 0.1

    # Training configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    max_steps: int = 1000
    block_size: int = 1024

    # T-SCU configuration
    power_budget_watts: float = 300.0
    target_efficiency: float = 1e-6  # bits per joule
    max_temperature_celsius: float = 85.0
    control_frequency: int = 10  # Apply control every N steps

    # Safety configuration
    emergency_power_limit: float = 400.0  # Absolute maximum power
    emergency_temperature_limit: float = 95.0  # Absolute maximum temperature

    # Logging configuration
    log_frequency: int = 10
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100


@dataclass
class TrainingState:
    """Current training state information"""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    power_consumption: float
    efficiency: float
    temperature: float
    control_factor: float
    timestamp: float


class PowerAwareTrainer:
    """
    Power-aware trainer that integrates T-SCU control into training loops.

    Key Features:
    - Real-time power monitoring and control
    - Dynamic learning rate adjustment based on efficiency
    - Thermal management during training
    - Comprehensive logging and analysis
    """

    def __init__(
        self,
        config: PowerAwareTrainingConfig,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        dataloader: Optional[DataLoader] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize power-aware trainer

        Args:
            config: Training configuration
            model: Pre-loaded model (optional, will load if None)
            tokenizer: Pre-loaded tokenizer (optional)
            dataloader: Training data loader (optional)
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or self._setup_default_logger()

        # Initialize model and tokenizer
        self.model = model or self._load_model()
        self.tokenizer = tokenizer or self._load_tokenizer()

        # Initialize data loader
        self.dataloader = dataloader

        # Initialize T-SCU components
        self.power_monitor = MultiDevicePowerMonitor()
        self.thermo_controller = ThermodynamicSCU(
            power_budget_watts=config.power_budget_watts,
            target_efficiency=config.target_efficiency,
            max_temperature_celsius=config.max_temperature_celsius
        )
        self.energy_metrics = EnergyEntropyMetrics()

        # Training state
        self.training_state = TrainingState(
            step=0, epoch=0, loss=0.0, learning_rate=config.learning_rate,
            power_consumption=0.0, efficiency=0.0, temperature=0.0,
            control_factor=1.0, timestamp=time.time()
        )

        # History for analysis
        self.training_history: List[TrainingState] = []
        self.power_readings_history: List[Dict] = []

        # Control state
        self.last_control_step = 0
        self.emergency_shutdown_triggered = False

        # Optimization states
        self.current_control_factor = 1.0
        self.adaptive_lr_scale = 1.0

        self.logger.info("PowerAwareTrainer initialized")

    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger("PowerAwareTrainer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_model(self) -> nn.Module:
        """Load and configure the model"""
        self.logger.info(f"Loading model: {self.config.model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Apply LoRA if specified
        if self.config.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_rank * 2,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, peft_config)
            self.logger.info("LoRA adapters applied")

        # Training optimizations
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False

        try:
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        except:
            pass

        return model

    def _load_tokenizer(self) -> Any:
        """Load tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def initialize_systems(self) -> bool:
        """Initialize all monitoring and control systems"""
        self.logger.info("Initializing T-SCU systems...")

        # Initialize power monitoring
        if not self.power_monitor.initialize():
            self.logger.error("Failed to initialize power monitoring")
            return False

        # Test power monitoring
        initial_readings = self.power_monitor.get_all_power_readings()
        if not initial_readings:
            self.logger.warning("No power readings available")
        else:
            total_power = sum(r.power_watts for r in initial_readings)
            self.logger.info(f"Initial power consumption: {total_power:.2f}W")

        self.logger.info("T-SCU systems initialized successfully")
        return True

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute a single training step with power awareness

        Args:
            batch: Training batch

        Returns:
            Training loss
        """
        # Apply control factor to batch size if needed
        if self.current_control_factor < 0.5:
            # Reduce effective batch size during power constraints
            batch_size = len(batch['input_ids'])
            new_batch_size = max(1, int(batch_size * self.current_control_factor))
            if new_batch_size < batch_size:
                for key in batch:
                    batch[key] = batch[key][:new_batch_size]

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss

        # Apply adaptive scaling based on efficiency
        scaled_loss = loss * self.adaptive_lr_scale

        # Backward pass
        scaled_loss.backward()

        return loss.item()

    def update_control_systems(self, step: int, loss: float) -> Dict[str, float]:
        """
        Update T-SCU control systems based on current state

        Args:
            step: Current training step
            loss: Current training loss

        Returns:
            Control information and adjustments
        """
        # Get current power readings
        power_readings = self.power_monitor.get_all_power_readings()
        total_power = sum(r.power_watts for r in power_readings)

        # Get average temperature
        temperatures = [r.temperature_celsius for r in power_readings if r.temperature_celsius]
        avg_temperature = max(temperatures) if temperatures else 25.0

        # Calculate information entropy of model parameters
        param_entropy = self.energy_metrics.calculate_parameter_entropy(self.model)

        # Create thermodynamic state
        thermo_state = self.thermo_controller.calculate_thermodynamic_metrics(
            information_entropy=param_entropy,
            power_consumption=total_power,
            temperature=avg_temperature + 273.15  # Convert to Kelvin
        )

        # Get control action
        control_factor, control_breakdown = self.thermo_controller.compute_control_action(
            thermo_state, step
        )

        # Apply control actions
        self.current_control_factor = control_factor

        # Emergency checks
        if total_power > self.config.emergency_power_limit:
            self.logger.warning(f"Emergency: Power {total_power:.1f}W exceeds limit {self.config.emergency_power_limit}W")
            self.current_control_factor = 0.3  # Aggressive throttling

        if avg_temperature > self.config.emergency_temperature_limit:
            self.logger.warning(f"Emergency: Temperature {avg_temperature:.1f}°C exceeds limit {self.config.emergency_temperature_limit}°C")
            self.current_control_factor = 0.2  # Very aggressive throttling
            self.emergency_shutdown_triggered = True

        # Adaptive learning rate adjustment
        self.adaptive_lr_scale = self._calculate_adaptive_lr_scale(
            thermo_state, control_breakdown
        )

        # Store control information
        control_info = {
            'power_consumption': total_power,
            'temperature': avg_temperature,
            'control_factor': control_factor,
            'adaptive_lr_scale': self.adaptive_lr_scale,
            'thermodynamic_state': thermo_state,
            'control_breakdown': control_breakdown
        }

        return control_info

    def _calculate_adaptive_lr_scale(
        self,
        thermo_state: ThermodynamicState,
        control_breakdown: Dict[str, float]
    ) -> float:
        """
        Calculate adaptive learning rate scaling based on thermodynamic efficiency

        Args:
            thermo_state: Current thermodynamic state
            control_breakdown: Control action breakdown

        Returns:
            Learning rate scaling factor
        """
        # Base scaling on efficiency
        efficiency_ratio = thermo_state.efficiency_bits_per_joule / self.config.target_efficiency

        # High efficiency -> can increase learning rate
        # Low efficiency -> decrease learning rate
        if efficiency_ratio > 1.2:
            lr_scale = 1.1  # Increase LR
        elif efficiency_ratio < 0.8:
            lr_scale = 0.9  # Decrease LR
        else:
            lr_scale = 1.0  # Keep LR unchanged

        # Consider temperature margin
        temp_margin = self.config.max_temperature_celsius - (thermo_state.temperature - 273.15)
        if temp_margin < 5:  # Less than 5°C margin
            lr_scale *= 0.8  # Reduce LR for thermal stability

        return max(0.5, min(1.5, lr_scale))

    def train(self, optimizer: torch.optim.Optimizer, scheduler: Any = None) -> Dict[str, Any]:
        """
        Main training loop with power awareness

        Args:
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)

        Returns:
            Training results and statistics
        """
        self.logger.info("Starting power-aware training...")

        if not self.initialize_systems():
            raise RuntimeError("Failed to initialize T-SCU systems")

        self.model.train()
        start_time = time.time()

        try:
            for step, batch in enumerate(self.dataloader):
                if step >= self.config.max_steps:
                    break

                if self.emergency_shutdown_triggered:
                    self.logger.error("Emergency shutdown triggered - stopping training")
                    break

                batch_start_time = time.time()

                # Training step
                loss = self.train_step(batch)

                # Update control systems
                if step % self.config.control_frequency == 0:
                    control_info = self.update_control_systems(step, loss)
                else:
                    control_info = {'power_consumption': 0, 'temperature': 0}

                # Gradient accumulation and optimization
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Apply adaptive learning rate scaling
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= self.adaptive_lr_scale

                    optimizer.step()
                    optimizer.zero_grad()

                    if scheduler:
                        scheduler.step()

                # Update training state
                self.training_state = TrainingState(
                    step=step,
                    epoch=step // len(self.dataloader),
                    loss=loss,
                    learning_rate=optimizer.param_groups[0]['lr'],
                    power_consumption=control_info.get('power_consumption', 0),
                    efficiency=control_info.get('thermodynamic_state', {}).get('efficiency_bits_per_joule', 0),
                    temperature=control_info.get('temperature', 0),
                    control_factor=self.current_control_factor,
                    timestamp=time.time()
                )

                # Log progress
                if step % self.config.log_frequency == 0:
                    self._log_progress(step, loss, control_info)

                # Save checkpoint
                if self.config.save_checkpoints and step % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(step, optimizer, loss)

                # Store history
                self.training_history.append(self.training_state)

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            # Cleanup
            self.power_monitor.shutdown()

        training_time = time.time() - start_time

        # Calculate final statistics
        results = self._calculate_training_statistics(training_time)

        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        return results

    def _log_progress(self, step: int, loss: float, control_info: Dict[str, Any]) -> None:
        """Log training progress"""
        power = control_info.get('power_consumption', 0)
        temp = control_info.get('temperature', 0)
        control_factor = control_info.get('control_factor', 1.0)

        self.logger.info(
            f"Step {step:4d} | Loss: {loss:.4f} | Power: {power:6.2f}W | "
            f"Temp: {temp:5.1f}°C | Control: {control_factor:.2f} | "
            f"LR Scale: {self.adaptive_lr_scale:.2f}"
        )

    def _save_checkpoint(self, step: int, optimizer: torch.optim.Optimizer, loss: float) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'training_state': self.training_state,
            'config': self.config
        }

        checkpoint_path = f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _calculate_training_statistics(self, training_time: float) -> Dict[str, Any]:
        """Calculate comprehensive training statistics"""
        if not self.training_history:
            return {"status": "no_data"}

        # Basic statistics
        total_steps = len(self.training_history)
        avg_loss = sum(state.loss for state in self.training_history) / total_steps
        avg_power = sum(state.power_consumption for state in self.training_history) / total_steps
        avg_temp = sum(state.temperature for state in self.training_history) / total_steps

        # Power statistics
        power_efficiency = avg_power / self.config.power_budget_watts * 100

        # T-SCU performance analysis
        tscu_analysis = self.thermo_controller.analyze_performance()

        # Energy consumption
        total_energy_joules = avg_power * training_time
        total_energy_kwh = total_energy_joules / (1000 * 3600)

        return {
            'training_time_seconds': training_time,
            'total_steps': total_steps,
            'average_loss': avg_loss,
            'average_power_watts': avg_power,
            'average_temperature_celsius': avg_temp,
            'power_budget_utilization_percent': power_efficiency,
            'total_energy_consumed_kwh': total_energy_kwh,
            'tscu_performance': tscu_analysis,
            'efficiency_recommendations': self.thermo_controller.get_optimization_recommendations(),
            'emergency_shutdown_triggered': self.emergency_shutdown_triggered
        }