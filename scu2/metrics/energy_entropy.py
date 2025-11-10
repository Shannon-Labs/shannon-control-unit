"""
Energy-Entropy Coupling Metrics for T-SCU

Advanced metrics that bridge Shannon information entropy with physical
thermodynamic entropy to quantify computational efficiency.

These metrics form the foundation for T-SCU's control systems and provide
deep insights into the physics of neural network training.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)


@dataclass
class EnergyEntropyMetrics:
    """Container for energy-entropy coupling metrics"""

    # Information metrics
    parameter_entropy: float          # bits
    data_entropy: float              # bits
    mutual_information: float        # bits

    # Energy metrics
    parameter_energy: float          # joules
    computation_energy: float        # joules
    landauer_minimum_energy: float   # joules

    # Efficiency metrics
    bits_per_joule: float           # bits/J
    joules_per_bit: float           # J/bit
    landauer_efficiency: float      # ratio to theoretical minimum

    # Thermodynamic metrics
    entropy_production_rate: float  # J/K·s
    heat_dissipation_rate: float    # J/s (watts)
    thermodynamic_efficiency: float # dimensionless


class InformationEntropyCalculator:
    """Calculate various information entropy metrics for neural networks"""

    @staticmethod
    def calculate_parameter_entropy(model: nn.Module, sigma: float = 0.01) -> float:
        """
        Calculate Shannon entropy of model parameters under Gaussian prior.

        Args:
            model: Neural network model
            sigma: Standard deviation of Gaussian prior

        Returns:
            Parameter entropy in bits
        """
        param_entropy = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Gaussian entropy: H = 0.5 * log(2πeσ²) in nats
                # Convert to bits by dividing by ln(2)
                gaussian_entropy = 0.5 * math.log(2 * math.pi * math.e * sigma**2) / math.log(2)
                param_entropy += gaussian_entropy * param.numel()
                param_count += param.numel()

        return param_entropy

    @staticmethod
    def calculate_activation_entropy(activations: torch.Tensor) -> float:
        """
        Calculate entropy of neuron activations.

        Args:
            activations: Activation tensor

        Returns:
            Activation entropy in bits
        """
        # Flatten activations
        flat_acts = activations.flatten().float()

        # Create histogram
        hist = torch.histc(flat_acts, bins=100, min=flat_acts.min(), max=flat_acts.max())
        hist = hist / hist.sum()  # Normalize to probabilities

        # Calculate entropy: H = -Σ p(x) log₂ p(x)
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * math.log2(p.item())

        return entropy * activations.numel()  # Scale by number of activations

    @staticmethod
    def calculate_mutual_information(
        x: torch.Tensor,
        y: torch.Tensor,
        bins: int = 50
    ) -> float:
        """
        Estimate mutual information between two variables.

        Args:
            x, y: Tensors
            bins: Number of bins for histogram

        Returns:
            Mutual information in bits
        """
        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(
            x.flatten().cpu().numpy(),
            y.flatten().cpu().numpy(),
            bins=bins
        )

        # Normalize to joint probability
        p_xy = hist_2d / hist_2d.sum()

        # Marginal probabilities
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)

        # Calculate mutual information
        mi = 0.0
        for i in range(p_xy.shape[0]):
            for j in range(p_xy.shape[1]):
                if p_xy[i, j] > 0 and p_x[i, 0] > 0 and p_y[0, j] > 0:
                    mi += p_xy[i, j] * math.log2(p_xy[i, j] / (p_x[i, 0] * p_y[0, j]))

        return mi


class EnergyCalculator:
    """Calculate energy consumption and thermodynamic metrics"""

    @staticmethod
    def calculate_computation_energy(
        power_watts: float,
        time_seconds: float
    ) -> float:
        """
        Calculate energy consumption from power and time.

        Args:
            power_watts: Power consumption in watts
            time_seconds: Time duration in seconds

        Returns:
            Energy consumption in joules
        """
        return power_watts * time_seconds

    @staticmethod
    def calculate_landauer_energy(
        information_bits: float,
        temperature_kelvin: float
    ) -> float:
        """
        Calculate theoretical minimum energy (Landauer limit).

        Args:
            information_bits: Amount of information in bits
            temperature_kelvin: Temperature in Kelvin

        Returns:
            Minimum energy in joules
        """
        return information_bits * k_B * temperature_kelvin * math.log(2)

    @staticmethod
    def calculate_entropy_production_rate(
        power_watts: float,
        temperature_kelvin: float
    ) -> float:
        """
        Calculate thermodynamic entropy production rate.

        Args:
            power_watts: Power consumption (heat flow rate)
            temperature_kelvin: Temperature

        Returns:
            Entropy production rate in J/K·s
        """
        return power_watts / temperature_kelvin


class EfficiencyAnalyzer:
    """Analyze computational efficiency from energy-entropy perspective"""

    @staticmethod
    def calculate_bits_per_joule(
        information_bits: float,
        energy_joules: float
    ) -> float:
        """Calculate information processing efficiency"""
        if energy_joules <= 0:
            return 0.0
        return information_bits / energy_joules

    @staticmethod
    def calculate_landauer_efficiency(
        actual_energy: float,
        landauer_minimum: float
    ) -> float:
        """Calculate efficiency relative to Landauer limit"""
        if landauer_minimum <= 0:
            return 0.0
        return landauer_minimum / actual_energy

    @staticmethod
    def calculate_thermodynamic_efficiency(
        useful_work_joules: float,
        total_energy_joules: float,
        temperature_hot: float,
        temperature_cold: float
    ) -> float:
        """
        Calculate thermodynamic efficiency (Carnot efficiency).

        Args:
            useful_work_joules: Useful work output
            total_energy_joules: Total energy input
            temperature_hot: Hot reservoir temperature (K)
            temperature_cold: Cold reservoir temperature (K)

        Returns:
            Thermodynamic efficiency (dimensionless)
        """
        if total_energy_joules <= 0:
            return 0.0

        actual_efficiency = useful_work_joules / total_energy_joules
        carnot_efficiency = 1 - (temperature_cold / temperature_hot)

        return min(actual_efficiency / carnot_efficiency, 1.0)


class AdvancedEnergyEntropyMetrics:
    """
    Advanced metrics that combine information theory with thermodynamics
    for deep analysis of neural network training efficiency.
    """

    def __init__(self):
        self.info_calc = InformationEntropyCalculator()
        self.energy_calc = EnergyCalculator()
        self.efficiency_analyzer = EfficiencyAnalyzer()

    def calculate_comprehensive_metrics(
        self,
        model: nn.Module,
        power_watts: float,
        temperature_kelvin: float,
        time_seconds: float = 1.0,
        loss_nats: float = 0.0,
        activations: Optional[torch.Tensor] = None
    ) -> EnergyEntropyMetrics:
        """
        Calculate comprehensive energy-entropy metrics.

        Args:
            model: Neural network model
            power_watts: Current power consumption
            temperature_kelvin: Current temperature
            time_seconds: Time period for calculation
            loss_nats: Cross-entropy loss in nats
            activations: Model activations (optional)

        Returns:
            Comprehensive energy-entropy metrics
        """
        # Information metrics
        param_entropy = self.info_calc.calculate_parameter_entropy(model)
        data_entropy = loss_nats / math.log(2) if loss_nats > 0 else 0.0

        # Activation entropy if available
        activation_entropy = 0.0
        if activations is not None:
            activation_entropy = self.info_calc.calculate_activation_entropy(activations)

        # Total information processed
        total_info_bits = param_entropy + data_entropy + activation_entropy

        # Energy metrics
        computation_energy = self.energy_calc.calculate_computation_energy(
            power_watts, time_seconds
        )
        landauer_energy = self.energy_calc.calculate_landauer_energy(
            total_info_bits, temperature_kelvin
        )

        # Efficiency metrics
        bits_per_joule = self.efficiency_analyzer.calculate_bits_per_joule(
            total_info_bits, computation_energy
        )
        joules_per_bit = 1.0 / bits_per_joule if bits_per_joule > 0 else float('inf')
        landauer_eff = self.efficiency_analyzer.calculate_landauer_efficiency(
            computation_energy, landauer_energy
        )

        # Thermodynamic metrics
        entropy_production_rate = self.energy_calc.calculate_entropy_production_rate(
            power_watts, temperature_kelvin
        )
        heat_dissipation_rate = power_watts  # All electrical power becomes heat

        # Thermodynamic efficiency (assume room temperature as cold reservoir)
        room_temp_kelvin = 298.15  # 25°C
        thermodynamic_eff = self.efficiency_analyzer.calculate_thermodynamic_efficiency(
            landauer_energy, computation_energy, temperature_kelvin, room_temp_kelvin
        )

        return EnergyEntropyMetrics(
            parameter_entropy=param_entropy,
            data_entropy=data_entropy,
            mutual_information=0.0,  # Would need input data to calculate
            parameter_energy=landauer_energy,
            computation_energy=computation_energy,
            landauer_minimum_energy=landauer_energy,
            bits_per_joule=bits_per_joule,
            joules_per_bit=joules_per_bit,
            landauer_efficiency=landauer_eff,
            entropy_production_rate=entropy_production_rate,
            heat_dissipation_rate=heat_dissipation_rate,
            thermodynamic_efficiency=thermodynamic_eff
        )

    def analyze_efficiency_bottlenecks(
        self,
        metrics: EnergyEntropyMetrics,
        target_efficiency: float = 1e-6
    ) -> List[str]:
        """
        Analyze efficiency bottlenecks and provide recommendations.

        Args:
            metrics: Current energy-entropy metrics
            target_efficiency: Target bits per joule efficiency

        Returns:
            List of bottleneck analysis and recommendations
        """
        bottlenecks = []

        # Check Landauer efficiency
        if metrics.landauer_efficiency < 1e-9:
            bottlenecks.append(
                f"Severe inefficiency: {metrics.landauer_efficiency:.2e} of Landauer limit. "
                "Consider hardware optimization or algorithmic improvements."
            )
        elif metrics.landauer_efficiency < 1e-7:
            bottlenecks.append(
                f"Low efficiency: {metrics.landauer_efficiency:.2e} of Landauer limit. "
                "Opportunity for significant optimization."
            )

        # Check bits per joule efficiency
        efficiency_ratio = metrics.bits_per_joule / target_efficiency
        if efficiency_ratio < 0.1:
            bottlenecks.append(
                f"Very low computational efficiency: {efficiency_ratio:.1%} of target. "
                "Major algorithmic or hardware changes needed."
            )
        elif efficiency_ratio < 0.5:
            bottlenecks.append(
                f"Below target efficiency: {efficiency_ratio:.1%} of target. "
                "Consider model architecture optimizations."
            )

        # Check entropy production
        if metrics.entropy_production_rate > 1.0:  # J/K·s
            bottlenecks.append(
                f"High entropy production: {metrics.entropy_production_rate:.3f} J/K·s. "
                "Focus on reducing irreversible operations."
            )

        # Check thermodynamic efficiency
        if metrics.thermodynamic_efficiency < 0.01:
            bottlenecks.append(
                f"Low thermodynamic efficiency: {metrics.thermodynamic_efficiency:.2%}. "
                "Consider reversible computing techniques or better heat management."
            )

        if not bottlenecks:
            bottlenecks.append("Efficiency looks good - no major bottlenecks detected.")

        return bottlenecks

    def calculate_improvement_potential(
        self,
        current_metrics: EnergyEntropyMetrics,
        target_metrics: Optional[EnergyEntropyMetrics] = None
    ) -> Dict[str, float]:
        """
        Calculate theoretical improvement potential.

        Args:
            current_metrics: Current performance metrics
            target_metrics: Target performance (optional)

        Returns:
            Dictionary of improvement potentials
        """
        if target_metrics is None:
            # Assume ideal Landauer efficiency as target
            target_metrics = EnergyEntropyMetrics(
                parameter_entropy=current_metrics.parameter_entropy,
                data_entropy=current_metrics.data_entropy,
                mutual_information=0.0,
                parameter_energy=current_metrics.landauer_minimum_energy,
                computation_energy=current_metrics.landauer_minimum_energy,
                landauer_minimum_energy=current_metrics.landauer_minimum_energy,
                bits_per_joule=1.0 / (k_B * 298.15 * math.log(2)),  # Theoretical maximum
                joules_per_bit=k_B * 298.15 * math.log(2),
                landauer_efficiency=1.0,
                entropy_production_rate=current_metrics.parameter_entropy / 298.15,
                heat_dissipation_rate=current_metrics.landauer_minimum_energy,
                thermodynamic_efficiency=1.0
            )

        # Calculate improvement ratios
        energy_reduction_potential = 1.0 - (target_metrics.computation_energy / current_metrics.computation_energy)
        efficiency_improvement_potential = (target_metrics.bits_per_joule / current_metrics.bits_per_joule) - 1.0
        landauer_improvement_potential = (target_metrics.landauer_efficiency / current_metrics.landauer_efficiency) - 1.0

        return {
            'energy_reduction_potential': energy_reduction_potential,
            'efficiency_improvement_potential': efficiency_improvement_potential,
            'landauer_improvement_potential': landauer_improvement_potential,
            'current_bits_per_joule': current_metrics.bits_per_joule,
            'target_bits_per_joule': target_metrics.bits_per_joule,
            'current_landauer_efficiency': current_metrics.landauer_efficiency,
            'target_landauer_efficiency': target_metrics.landauer_efficiency
        }