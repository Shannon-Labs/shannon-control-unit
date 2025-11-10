"""
Multi-Scale Entropy Analysis with Dyadic Wavelet Filter Banks

Revolutionary entropy analysis that goes beyond fixed micro/meso/macro scales
by using learned dyadic wavelet decompositions to identify significant scales
from the data itself.

Uses Daubechies 4-tap wavelets for optimal time-frequency localization
and energy-based scale selection to keep only computationally relevant scales.
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import signal
from scipy.fft import fft, ifft
import torch
import torch.nn.functional as F

# Physical constants for Landauer calculations
k_B = 1.380649e-23  # Boltzmann constant (J/K)
LANDAUER_ENERGY_BIT_300K = k_B * 300 * math.log(2)  # ~2.8e-21 J per bit


@dataclass
class WaveletScale:
    """Information about a discovered wavelet scale"""
    scale_index: int
    frequency_band: Tuple[float, float]  # Hz
    energy_ratio: float                  # Energy / total energy
    entropy_bits: float                  # Shannon entropy at this scale
    landauer_energy_joules: float        # Theoretical minimum energy
    significance_score: float            # Information significance
    wavelet_coeffs: np.ndarray          # Wavelet coefficients


@dataclass
class MultiScaleEntropyResult:
    """Complete multi-scale entropy analysis result"""
    total_entropy_bits: float
    landauer_residual_bits: float       # Zero-scale Landauer residual
    selected_scales: List[WaveletScale]
    scale_mask_64bit: int               # 64-bit mask of selected scales
    efficiency_gap: float               # (E_measured - E_Landauer) / E_Landauer
    temperature_kelvin: float
    timestamp: float


class DaubechiesWaveletBank:
    """
    Daubechies 4-tap wavelet filter bank for multi-scale analysis.

    Uses dyadic decomposition (powers of 2) to analyze entropy at
    multiple time scales, selecting only the most significant ones.
    """

    def __init__(self, max_scales: int = 8, energy_threshold: float = 0.05):
        """
        Initialize wavelet filter bank.

        Args:
            max_scales: Maximum number of dyadic scales to analyze
            energy_threshold: Minimum energy ratio to keep a scale (default 5%)
        """
        self.max_scales = max_scales
        self.energy_threshold = energy_threshold

        # Daubechies 4-tap wavelet coefficients
        self._setup_daubechies_4tap()

        # Learned scale mask (starts with all scales selected)
        self.scale_mask = np.ones(max_scales, dtype=bool)
        self.scale_mask_64bit = (1 << max_scales) - 1  # All scales initially selected

        # EMA update parameters for online learning
        self.ema_alpha = 0.1  # Learning rate for scale energy updates
        self.scale_energy_history = []

    def _setup_daubechies_4tap(self):
        """Setup Daubechies 4-tap wavelet filters"""
        # Daubechies 4 (db4) scaling filter coefficients
        h = np.array([
            0.03222310060464282,
            -0.01260396726203744,
            -0.09921954360302915,
            0.29785779560527736,
            0.8037387518049192,
            0.49761866763201545,
            -0.0956472612644853,
            -0.07111391863189652,
            0.02163056967207876,
            0.007092100109093673
        ])

        # Wavelet filter (high-pass) is the quadrature mirror
        g = ((-1) ** np.arange(len(h))) * h[::-1]

        self.scaling_filter = h
        self.wavelet_filter = g

    def analyze_signal(self, signal_data: np.ndarray, temperature_kelvin: float = 300.0) -> MultiScaleEntropyResult:
        """
        Perform multi-scale entropy analysis on input signal.

        Args:
            signal_data: Input signal (e.g., gradient norms, loss values, etc.)
            temperature_kelvin: Current temperature for Landauer calculations

        Returns:
            Complete multi-scale entropy analysis result
        """
        # Ensure signal length is sufficient for decomposition
        min_length = 2 ** (self.max_scales + 2)
        if len(signal_data) < min_length:
            # Pad signal to minimum length
            pad_length = min_length - len(signal_data)
            signal_data = np.pad(signal_data, (0, pad_length), mode='constant')

        # Initialize analysis
        approximation = signal_data.copy()
        scales = []
        total_energy = np.sum(signal_data ** 2)

        # Dyadic wavelet decomposition
        for scale_idx in range(self.max_scales):
            if len(approximation) < 4:  # Need at least 4 samples for db4
                break

            # Apply wavelet transform
            details, approximation = self._wavelet_decompose(approximation)

            # Calculate energy and entropy at this scale
            energy = np.sum(details ** 2)
            energy_ratio = energy / total_energy if total_energy > 0 else 0

            # Calculate Shannon entropy of wavelet coefficients
            entropy_bits = self._calculate_coefficient_entropy(details)

            # Calculate Landauer energy for this scale
            landauer_energy = entropy_bits * LANDAUER_ENERGY_BIT_300K * (temperature_kelvin / 300.0)

            # Calculate significance score (combination of energy and information content)
            significance_score = energy_ratio * (1 + np.log2(1 + entropy_bits))

            scale = WaveletScale(
                scale_index=scale_idx,
                frequency_band=self._get_frequency_band(scale_idx, len(signal_data)),
                energy_ratio=energy_ratio,
                entropy_bits=entropy_bits,
                landauer_energy_joules=landauer_energy,
                significance_score=significance_score,
                wavelet_coeffs=details
            )

            scales.append(scale)

        # Update scale mask based on energy (online learning)
        self._update_scale_mask(scales)

        # Select only significant scales
        selected_scales = [s for i, s in enumerate(scales) if self.scale_mask[i]]

        # Calculate Landauer residual (zero-scale)
        landauer_residual_bits = self._calculate_landauer_residual(signal_data, selected_scales, temperature_kelvin)

        # Calculate total entropy
        total_entropy = landauer_residual_bits + sum(s.entropy_bits for s in selected_scales)

        # Calculate efficiency gap
        total_measured_energy = np.sum(signal_data ** 2) * 1e-12  # Rough conversion to joules
        total_landauer_energy = total_entropy * LANDAUER_ENERGY_BIT_300K * (temperature_kelvin / 300.0)
        efficiency_gap = (total_measured_energy - total_landauer_energy) / total_landauer_energy if total_landauer_energy > 0 else 0

        return MultiScaleEntropyResult(
            total_entropy_bits=total_entropy,
            landauer_residual_bits=landauer_residual_bits,
            selected_scales=selected_scales,
            scale_mask_64bit=self.scale_mask_64bit,
            efficiency_gap=efficiency_gap,
            temperature_kelvin=temperature_kelvin,
            timestamp=time.time()
        )

    def _wavelet_decompose(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one level of wavelet decomposition using Daubechies 4-tap"""
        # Convolution with wavelet and scaling filters
        n = len(signal_data)

        # Periodic extension for convolution
        padded_signal = np.pad(signal_data, (len(self.wavelet_filter) - 1, 0), mode='wrap')

        # Detail coefficients (high-pass)
        details = signal.convolve(padded_signal, self.wavelet_filter, mode='valid')[:n//2]

        # Approximation coefficients (low-pass)
        approximation = signal.convolve(padded_signal, self.scaling_filter, mode='valid')[:n//2]

        # Downsample by 2 (dyadic decomposition)
        details = details[::2]
        approximation = approximation[::2]

        return details, approximation

    def _calculate_coefficient_entropy(self, coefficients: np.ndarray) -> float:
        """Calculate Shannon entropy of wavelet coefficients"""
        if len(coefficients) == 0:
            return 0.0

        # Normalize coefficients to create probability distribution
        coeffs_abs = np.abs(coefficients)
        total = np.sum(coeffs_abs)

        if total == 0:
            return 0.0

        probs = coeffs_abs / total

        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Add small epsilon to avoid log(0)

        return entropy

    def _get_frequency_band(self, scale_idx: int, signal_length: int) -> Tuple[float, float]:
        """Get frequency band for a given scale"""
        # Nyquist frequency
        nyquist = 0.5

        # Dyadic decomposition halves the frequency band at each scale
        freq_high = nyquist / (2 ** scale_idx)
        freq_low = nyquist / (2 ** (scale_idx + 1))

        return (freq_low, freq_high)

    def _update_scale_mask(self, scales: List[WaveletScale]):
        """Update scale mask using EMA based on energy ratios"""
        if not scales:
            return

        # Extract energy ratios
        current_energies = np.array([s.energy_ratio for s in scales])

        # Update energy history with EMA
        if not self.scale_energy_history:
            self.scale_energy_history = current_energies
        else:
            self.scale_energy_history = (self.ema_alpha * current_energies +
                                       (1 - self.ema_alpha) * self.scale_energy_history)

        # Update mask based on average energy
        for i, avg_energy in enumerate(self.scale_energy_history):
            if avg_energy < self.energy_threshold:
                self.scale_mask[i] = False
            else:
                self.scale_mask[i] = True

        # Update 64-bit mask
        self.scale_mask_64bit = 0
        for i, selected in enumerate(self.scale_mask):
            if selected:
                self.scale_mask_64bit |= (1 << i)

    def _calculate_landauer_residual(self, signal_data: np.ndarray, selected_scales: List[WaveletScale], temperature_kelvin: float) -> float:
        """
        Calculate Landauer residual (zero-scale entropy).

        This represents the theoretically unavoidable information processing
        that cannot be captured by the wavelet decomposition.
        """
        # Calculate total variance in signal
        total_variance = np.var(signal_data)

        # Sum variance explained by selected scales
        explained_variance = sum(np.var(s.wavelet_coeffs) for s in selected_scales)

        # Residual variance (unexplained)
        residual_variance = max(0, total_variance - explained_variance)

        # Convert residual variance to information bits
        if residual_variance > 0:
            # Using Gaussian entropy formula: H = 0.5 * log2(2πeσ²)
            residual_bits = 0.5 * np.log2(2 * np.pi * np.e * residual_variance)
        else:
            residual_bits = 0.0

        # Ensure minimum Landauer residual (at least 1 bit of information)
        residual_bits = max(1.0, residual_bits)

        return residual_bits

    def get_scale_summary(self, result: MultiScaleEntropyResult) -> Dict:
        """Get summary statistics of multi-scale analysis"""
        summary = {
            'total_scales_analyzed': self.max_scales,
            'selected_scales_count': len(result.selected_scales),
            'scale_mask_hex': hex(result.scale_mask_64bit),
            'landauer_residual_bits': result.landauer_residual_bits,
            'total_entropy_bits': result.total_entropy_bits,
            'efficiency_gap': result.efficiency_gap,
            'scales': []
        }

        for scale in result.selected_scales:
            scale_info = {
                'scale': scale.scale_index,
                'frequency_band_hz': scale.frequency_band,
                'energy_ratio': scale.energy_ratio,
                'entropy_bits': scale.entropy_bits,
                'landauer_energy_joules': scale.landauer_energy_joules,
                'significance_score': scale.significance_score
            }
            summary['scales'].append(scale_info)

        return summary

    def save_scale_mask(self, filepath: str):
        """Save learned scale mask to file for reproducibility"""
        mask_data = {
            'scale_mask_64bit': self.scale_mask_64bit,
            'scale_mask': self.scale_mask.tolist(),
            'max_scales': self.max_scales,
            'energy_threshold': self.energy_threshold,
            'ema_alpha': self.ema_alpha,
            'timestamp': time.time()
        }

        np.save(filepath, mask_data)
        print(f"Scale mask saved to {filepath}")

    def load_scale_mask(self, filepath: str) -> bool:
        """Load learned scale mask from file"""
        try:
            mask_data = np.load(filepath, allow_pickle=True).item()

            self.scale_mask_64bit = mask_data['scale_mask_64bit']
            self.scale_mask = np.array(mask_data['scale_mask'], dtype=bool)
            self.max_scales = mask_data['max_scales']
            self.energy_threshold = mask_data['energy_threshold']
            self.ema_alpha = mask_data['ema_alpha']

            print(f"Scale mask loaded from {filepath}")
            return True

        except Exception as e:
            print(f"Failed to load scale mask: {e}")
            return False


class MultiScaleEntropyAnalyzer:
    """
    High-level interface for multi-scale entropy analysis in T-SCU.

    Integrates with the thermodynamic controller to provide adaptive
    entropy-based control at multiple time scales.
    """

    def __init__(self, max_scales: int = 8, energy_threshold: float = 0.05):
        self.wavelet_bank = DaubechiesWaveletBank(max_scales, energy_threshold)
        self.analysis_history = []

    def analyze_training_signal(self,
                               loss_history: List[float],
                               gradient_norms: List[float],
                               temperature_kelvin: float = 300.0) -> MultiScaleEntropyResult:
        """
        Analyze training signals for multi-scale entropy patterns.

        Args:
            loss_history: Recent training loss values
            gradient_norms: Recent gradient norm values
            temperature_kelvin: Current temperature

        Returns:
            Multi-scale entropy analysis result
        """
        # Combine loss and gradient signals
        if len(loss_history) != len(gradient_norms):
            # Pad shorter signal
            min_len = min(len(loss_history), len(gradient_norms))
            loss_history = loss_history[-min_len:]
            gradient_norms = gradient_norms[-min_len:]

        # Normalize signals
        loss_signal = np.array(loss_history)
        loss_signal = (loss_signal - np.mean(loss_signal)) / (np.std(loss_signal) + 1e-8)

        grad_signal = np.array(gradient_norms)
        grad_signal = (grad_signal - np.mean(grad_signal)) / (np.std(grad_signal) + 1e-8)

        # Combine signals (weighted sum)
        combined_signal = 0.6 * loss_signal + 0.4 * grad_signal

        # Perform multi-scale analysis
        result = self.wavelet_bank.analyze_signal(combined_signal, temperature_kelvin)

        # Store in history
        self.analysis_history.append(result)
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-500:]  # Keep last 500 analyses

        return result

    def get_adaptive_control_params(self, result: MultiScaleEntropyResult) -> Dict[str, float]:
        """
        Get adaptive control parameters based on multi-scale entropy analysis.

        Args:
            result: Multi-scale entropy analysis result

        Returns:
            Dictionary of control parameters for T-SCU
        """
        params = {
            'entropy_control_factor': 1.0,
            'scale_specific_adjustments': {},
            'landauer_penalty': 0.0,
            'efficiency_target_adjustment': 1.0
        }

        # Adjust control based on entropy distribution
        if result.selected_scales:
            # High entropy at fine scales suggests need for more regularization
            fine_scale_entropy = sum(s.entropy_bits for s in result.selected_scales[:3])
            total_scale_entropy = sum(s.entropy_bits for s in result.selected_scales)

            if total_scale_entropy > 0:
                fine_scale_ratio = fine_scale_scale_entropy / total_scale_entropy
                if fine_scale_ratio > 0.6:  # Too much fine-scale entropy
                    params['entropy_control_factor'] = 0.8  # Reduce learning rate
                elif fine_scale_ratio < 0.2:  # Too little fine-scale entropy
                    params['entropy_control_factor'] = 1.2  # Increase learning rate

        # Landauer residual penalty
        if result.landauer_residual_bits > result.total_entropy_bits * 0.5:
            params['landauer_penalty'] = 0.1  # Penalize inefficient computation

        # Efficiency gap adjustment
        if result.efficiency_gap > 10.0:  # Very inefficient
            params['efficiency_target_adjustment'] = 0.7  # Reduce efficiency target
        elif result.efficiency_gap < 1.0:  # Very efficient
            params['efficiency_target_adjustment'] = 1.3  # Increase efficiency target

        # Scale-specific adjustments
        for scale in result.selected_scales:
            if scale.significance_score > 0.1:  # Significant scale
                params['scale_specific_adjustments'][f'scale_{scale.scale_index}'] = scale.significance_score

        return params


# Convenience function for quick analysis
def quick_multiscale_analysis(signal_data: np.ndarray,
                             temperature_kelvin: float = 300.0,
                             max_scales: int = 8,
                             energy_threshold: float = 0.05) -> Dict:
    """
    Quick multi-scale entropy analysis for testing and debugging.

    Args:
        signal_data: Input signal
        temperature_kelvin: Temperature for Landauer calculations
        max_scales: Maximum scales to analyze
        energy_threshold: Energy threshold for scale selection

    Returns:
        Dictionary with analysis results
    """
    analyzer = MultiScaleEntropyAnalyzer(max_scales, energy_threshold)
    result = analyzer.wavelet_bank.analyze_signal(signal_data, temperature_kelvin)
    return analyzer.wavelet_bank.get_scale_summary(result)


if __name__ == "__main__":
    # Test the multi-scale entropy analyzer
    print("Testing Multi-Scale Entropy Analyzer...")

    # Generate test signal with multi-scale structure
    t = np.linspace(0, 10, 1000)
    test_signal = (
        np.sin(2 * np.pi * 1 * t) +           # Low frequency component
        0.5 * np.sin(2 * np.pi * 10 * t) +    # Medium frequency component
        0.2 * np.sin(2 * np.pi * 50 * t) +    # High frequency component
        0.1 * np.random.randn(len(t))         # Noise
    )

    # Analyze signal
    result = quick_multiscale_analysis(test_signal, temperature_kelvin=350.0)

    print(f"Multi-scale analysis results:")
    print(f"  Total entropy bits: {result['total_entropy_bits']:.2f}")
    print(f"  Landauer residual bits: {result['landauer_residual_bits']:.2f}")
    print(f"  Selected scales: {result['selected_scales_count']}/{result['total_scales_analyzed']}")
    print(f"  Scale mask: {result['scale_mask_hex']}")
    print(f"  Efficiency gap: {result['efficiency_gap']:.2f}")

    print("\nScale details:")
    for scale in result['scales']:
        print(f"  Scale {scale['scale']}: {scale['entropy_bits']:.2f} bits, "
              f"Energy ratio: {scale['energy_ratio']:.3f}, "
              f"Band: {scale['frequency_band_hz'][0]:.1f}-{scale['frequency_band_hz'][1]:.1f} Hz")

    print("\nMulti-scale entropy analyzer test completed successfully!")