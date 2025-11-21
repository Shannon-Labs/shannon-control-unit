"""
Apple Silicon Power Monitoring for T-SCU

Specialized power monitoring for Apple Silicon Macs using native macOS APIs.
Apple Silicon (M1/M2/M3) is actually ideal for T-SCU because:

1. Excellent built-in power monitoring capabilities
2. Industry-leading energy efficiency (close to Landauer limits)
3. Unified memory architecture reduces energy waste
4. Advanced thermal management systems
5. Native support for high-resolution power metrics
"""

import subprocess
import time
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import psutil

from .power_monitor import PowerReading, PowerMonitor


@dataclass
class ApplePowerReading:
    """Apple-specific power reading with additional metrics"""
    cpu_power_watts: float
    gpu_power_watts: float
    neural_engine_power_watts: float
    memory_power_watts: float
    total_power_watts: float
    cpu_temperature_celsius: float
    gpu_temperature_celsius: float
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    memory_bandwidth_gbps: float
    frequency_ghz: float


class AppleSiliconPowerMonitor(PowerMonitor):
    """
    Apple Silicon power monitoring using native macOS tools.

    Uses multiple approaches:
    1. powermetrics - Apple's native power analysis tool
    2. ioreg - Hardware registry access
    3. psutil - System resource monitoring
    4. sysctl - Kernel parameter access
    """

    def __init__(self):
        self.initialized = False
        self.platform = None
        self.chip_model = None
        self.last_reading_time = 0
        self.cached_readings = {}
        self.powermetrics_available = False

    def initialize(self) -> bool:
        """Initialize Apple Silicon power monitoring"""
        try:
            # Detect Apple Silicon
            self._detect_apple_silicon()
            if not self._is_apple_silicon():
                print("Warning: Not running on Apple Silicon")
                return False

            # Check if powermetrics is available (requires admin access)
            self.powermetrics_available = self._check_powermetrics_access()
            if not self.powermetrics_available:
                print("Warning: powermetrics requires admin access. Using fallback methods.")

            # Test power monitoring
            test_reading = self._get_power_reading()
            if test_reading and test_reading.total_power_watts > 0:
                self.initialized = True
                print(f"Apple Silicon T-SCU monitor initialized for {self.chip_model}")
                print(f"Power consumption: {test_reading.total_power_watts:.2f}W")
                return True
            else:
                print("Failed to get power readings")
                return False

        except Exception as e:
            print(f"Failed to initialize Apple power monitor: {e}")
            return False

    def _detect_apple_silicon(self) -> None:
        """Detect Apple Silicon chip model"""
        try:
            # Get chip model from system
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=5)
            self.chip_model = result.stdout.strip()
        except:
            self.chip_model = "Unknown Apple Silicon"

        # Check platform
        try:
            import platform
            self.platform = platform.platform()
        except:
            self.platform = "macOS"

    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon"""
        if not self.chip_model:
            return False
        return any(chip in self.chip_model for chip in ["Apple M1", "Apple M2", "Apple M3", "Apple M4"])

    def _check_powermetrics_access(self) -> bool:
        """Check if powermetrics tool is accessible"""
        try:
            # Quick test run (requires admin)
            result = subprocess.run([
                'sudo', 'powermetrics',
                '--samplers', 'cpu_power,gpu_power',
                '-i', '100',  # 100ms sample
                '-n', '1'     # 1 sample
            ], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def get_power_readings(self) -> List[PowerReading]:
        """Get power readings from Apple Silicon"""
        if not self.initialized:
            return []

        try:
            apple_reading = self._get_power_reading()
            if not apple_reading:
                return []

            timestamp = time.time()
            readings = []

            # CPU reading
            readings.append(PowerReading(
                timestamp=timestamp,
                component="cpu",
                identifier="apple_silicon_cpu",
                power_watts=apple_reading.cpu_power_watts,
                temperature_celsius=apple_reading.cpu_temperature_celsius,
                utilization_percent=apple_reading.cpu_utilization_percent,
                clock_mhz=apple_reading.frequency_ghz * 1000
            ))

            # GPU reading
            readings.append(PowerReading(
                timestamp=timestamp,
                component="gpu",
                identifier="apple_silicon_gpu",
                power_watts=apple_reading.gpu_power_watts,
                temperature_celsius=apple_reading.gpu_temperature_celsius,
                utilization_percent=apple_reading.gpu_utilization_percent
            ))

            # Neural Engine reading (Apple Silicon specific!)
            if apple_reading.neural_engine_power_watts > 0:
                readings.append(PowerReading(
                    timestamp=timestamp,
                    component="neural_engine",
                    identifier="apple_neural_engine",
                    power_watts=apple_reading.neural_engine_power_watts
                ))

            # Memory reading
            readings.append(PowerReading(
                timestamp=timestamp,
                component="memory",
                identifier="unified_memory",
                power_watts=apple_reading.memory_power_watts
            ))

            # Total system reading
            readings.append(PowerReading(
                timestamp=timestamp,
                component="system",
                identifier="total",
                power_watts=apple_reading.total_power_watts
            ))

            return readings

        except Exception as e:
            print(f"Error getting Apple power readings: {e}")
            return []

    def _get_power_reading(self) -> Optional[ApplePowerReading]:
        """Get comprehensive power reading"""
        if self.powermetrics_available:
            return self._get_powermetrics_reading()
        else:
            return self._get_fallback_reading()

    def _get_powermetrics_reading(self) -> Optional[ApplePowerReading]:
        """Get reading using powermetrics (requires admin access)"""
        try:
            # Run powermetrics with comprehensive samplers
            result = subprocess.run([
                'sudo', 'powermetrics',
                '--samplers', 'cpu_power,gpu_power,ane_power,disk_power,thermal',
                '-i', '200',  # 200ms sample
                '-n', '1'     # 1 sample
            ], capture_output=True, text=True, timeout=10)

            return self._parse_powermetrics_output(result.stdout)
        except Exception as e:
            print(f"powermetrics failed: {e}")
            return None

    def _parse_powermetrics_output(self, output: str) -> Optional[ApplePowerReading]:
        """Parse powermetrics output"""
        try:
            reading = ApplePowerReading(
                cpu_power_watts=0.0,
                gpu_power_watts=0.0,
                neural_engine_power_watts=0.0,
                memory_power_watts=0.0,
                total_power_watts=0.0,
                cpu_temperature_celsius=0.0,
                gpu_temperature_celsius=0.0,
                cpu_utilization_percent=0.0,
                gpu_utilization_percent=0.0,
                memory_bandwidth_gbps=0.0,
                frequency_ghz=0.0
            )

            # Parse CPU power
            cpu_power_match = re.search(r'CPU Power:\s*([\d.]+)\s*mW', output)
            if cpu_power_match:
                reading.cpu_power_watts = float(cpu_power_match.group(1)) / 1000

            # Parse GPU power
            gpu_power_match = re.search(r'GPU Power:\s*([\d.]+)\s*mW', output)
            if gpu_power_match:
                reading.gpu_power_watts = float(gpu_power_match.group(1)) / 1000

            # Parse Neural Engine (ANE) power
            ane_power_match = re.search(r'ANE Power:\s*([\d.]+)\s*mW', output)
            if ane_power_match:
                reading.neural_engine_power_watts = float(ane_power_match.group(1)) / 1000

            # Parse temperatures
            cpu_temp_match = re.search(r'CPU Core temperature:\s*([\d.]+)\s*C', output)
            if cpu_temp_match:
                reading.cpu_temperature_celsius = float(cpu_temp_match.group(1))

            gpu_temp_match = re.search(r'GPU temperature:\s*([\d.]+)\s*C', output)
            if gpu_temp_match:
                reading.gpu_temperature_celsius = float(gpu_temp_match.group(1))

            # Parse CPU frequency
            freq_match = re.search(r'CPU frequency:\s*([\d.]+)\s*MHz', output)
            if freq_match:
                reading.frequency_ghz = float(freq_match.group(1)) / 1000

            # Calculate total power (rough estimate)
            reading.total_power_watts = (
                reading.cpu_power_watts +
                reading.gpu_power_watts +
                reading.neural_engine_power_watts
            )

            # Estimate memory power based on bandwidth (Apple UMA is very efficient)
            reading.memory_power_watts = reading.total_power_watts * 0.1  # ~10% of total

            return reading

        except Exception as e:
            print(f"Error parsing powermetrics output: {e}")
            return None

    def _get_fallback_reading(self) -> Optional[ApplePowerReading]:
        """Get reading using alternative methods (no admin required)"""
        try:
            reading = ApplePowerReading(
                cpu_power_watts=0.0,
                gpu_power_watts=0.0,
                neural_engine_power_watts=0.0,
                memory_power_watts=0.0,
                total_power_watts=0.0,
                cpu_temperature_celsius=0.0,
                gpu_temperature_celsius=0.0,
                cpu_utilization_percent=0.0,
                gpu_utilization_percent=0.0,
                memory_bandwidth_gbps=0.0,
                frequency_ghz=0.0
            )

            # Get CPU utilization from psutil
            reading.cpu_utilization_percent = psutil.cpu_percent(interval=0.1)

            # Get memory info
            memory = psutil.virtual_memory()
            reading.memory_bandwidth_gbps = (memory.total - memory.available) / (1024**3) * 0.1  # Rough estimate

            # Estimate power consumption based on utilization
            # Apple Silicon is very efficient - these are conservative estimates
            if "M1" in self.chip_model:
                max_cpu_power = 20.0  # M1 Max CPU power
                max_gpu_power = 25.0  # M1 Max GPU power
            elif "M2" in self.chip_model:
                max_cpu_power = 25.0  # M2 Max CPU power
                max_gpu_power = 35.0  # M2 Max GPU power
            elif "M3" in self.chip_model:
                max_cpu_power = 30.0  # M3 Max CPU power
                max_gpu_power = 40.0  # M3 Max GPU power
            else:
                max_cpu_power = 25.0  # Conservative estimate
                max_gpu_power = 30.0  # Conservative estimate

            # Power scales roughly with utilization (non-linear)
            reading.cpu_power_watts = max_cpu_power * (reading.cpu_utilization_percent / 100) ** 1.3

            # Try to get temperature from ioreg
            reading.cpu_temperature_celsius = self._get_temperature_ioreg()

            # Estimate total power
            reading.total_power_watts = reading.cpu_power_watts * 1.5  # Include other components

            return reading

        except Exception as e:
            print(f"Error in fallback reading: {e}")
            return None

    def _get_temperature_ioreg(self) -> float:
        """Get temperature using ioreg (no admin required)"""
        try:
            result = subprocess.run([
                'ioreg', '-c', 'IOPlatformSensor', '-r', '-k', 'temperature'
            ], capture_output=True, text=True, timeout=5)

            # Parse temperature from ioreg output
            for line in result.stdout.split('\n'):
                if 'temperature' in line:
                    # Look for temperature values
                    temp_match = re.search(r'"([^"]+)"\s*=\s*([\d.]+)', line)
                    if temp_match and 'CPU' in temp_match.group(1):
                        temp_value = float(temp_match.group(2))
                        # Some Apple sensors report in different units
                        if temp_value > 100:  # Likely in Fahrenheit or scaled
                            temp_value = (temp_value - 32) * 5/9  # Convert from F to C
                        elif temp_value > 50:  # Likely already in Celsius
                            pass
                        else:  # Likely in some other unit, skip
                            continue
                        return temp_value

            return 45.0  # Default estimate for Apple Silicon

        except:
            return 45.0  # Default estimate

    def get_apple_specific_metrics(self) -> Dict[str, float]:
        """Get Apple Silicon specific efficiency metrics"""
        try:
            metrics = {
                'efficiency_score': 0.0,
                'memory_efficiency': 0.0,
                'neural_engine_utilization': 0.0,
                'thermal_headroom': 0.0
            }

            # Get power readings
            readings = self.get_power_readings()
            if not readings:
                return metrics

            # Calculate efficiency score (higher is better)
            cpu_reading = next((r for r in readings if r.component == "cpu"), None)
            if cpu_reading and cpu_reading.power_watts > 0:
                # Efficiency = performance per watt
                performance_score = cpu_reading.utilization_percent or 0
                metrics['efficiency_score'] = performance_score / cpu_reading.power_watts

            # Apple Silicon has excellent thermal characteristics
            cpu_temp = next((r.temperature_celsius for r in readings if r.component == "cpu"), 45.0)
            metrics['thermal_headroom'] = max(0, 85 - cpu_temp)  # Headroom to 85°C limit

            return metrics

        except Exception as e:
            print(f"Error getting Apple metrics: {e}")
            return {}

    def shutdown(self) -> None:
        """Clean shutdown"""
        self.initialized = False


def create_apple_power_monitor() -> AppleSiliconPowerMonitor:
    """Create Apple Silicon power monitor"""
    monitor = AppleSiliconPowerMonitor()
    monitor.initialize()
    return monitor


# Quick test function
def test_apple_power_monitor():
    """Test Apple Silicon power monitoring"""
    print("Testing Apple Silicon T-SCU Power Monitor...")

    monitor = create_apple_power_monitor()
    if not monitor.initialized:
        print("Failed to initialize Apple power monitor")
        return False

    print(f"Apple Silicon detected: {monitor.chip_model}")
    print(f"Powermetrics available: {monitor.powermetrics_available}")

    # Test readings
    for i in range(3):
        readings = monitor.get_power_readings()
        if readings:
            total_power = sum(r.power_watts for r in readings)
            print(f"Reading {i+1}: Total power = {total_power:.2f}W")

            for reading in readings:
                print(f"  {reading.component}: {reading.power_watts:.2f}W "
                      f"(Temp: {reading.temperature_celsius:.1f}°C)")
        time.sleep(1)

    # Get Apple-specific metrics
    metrics = monitor.get_apple_specific_metrics()
    print(f"Apple metrics: {metrics}")

    monitor.shutdown()
    print("Apple Silicon power monitor test completed")
    return True


if __name__ == "__main__":
    test_apple_power_monitor()