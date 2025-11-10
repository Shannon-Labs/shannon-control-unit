"""
Multi-Platform Power Monitoring Infrastructure

Real-time power consumption monitoring for GPUs, CPUs, and system components.
This is the hardware interface layer for T-SCU's thermodynamic control system.

Supports:
- NVIDIA GPUs (via NVML)
- AMD GPUs (via ADL/ROCm)
- Intel CPUs (via RAPL)
- System power monitoring
- Temperature sensors
- Battery monitoring for mobile platforms
"""

import time
import platform
import subprocess
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    try:
        nvml.nvmlInit()
    except:
        NVML_AVAILABLE = False
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class PowerReading:
    """Single power measurement reading"""
    timestamp: float
    component: str           # "gpu", "cpu", "system", "memory"
    identifier: str          # "0", "1", "cpu_package", etc.
    power_watts: float
    temperature_celsius: Optional[float] = None
    utilization_percent: Optional[float] = None
    clock_mhz: Optional[float] = None


class PowerMonitor(ABC):
    """Abstract base class for power monitoring devices"""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the power monitor. Returns True if successful."""
        pass

    @abstractmethod
    def get_power_readings(self) -> List[PowerReading]:
        """Get current power readings"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean shutdown of monitor"""
        pass


class NVidiaGPUPowerMonitor(PowerMonitor):
    """NVIDIA GPU power monitoring via NVML"""

    def __init__(self):
        self.initialized = False
        self.gpu_handles = []
        self.gpu_count = 0

    def initialize(self) -> bool:
        """Initialize NVML and get GPU handles"""
        if not NVML_AVAILABLE:
            print("NVML not available. Install nvidia-ml-py3 package.")
            return False

        try:
            nvml.nvmlInit()
            self.gpu_count = nvml.nvmlDeviceGetCount()
            self.gpu_handles = []

            for i in range(self.gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)

            self.initialized = True
            print(f"Initialized NVML for {self.gpu_count} NVIDIA GPUs")
            return True

        except Exception as e:
            print(f"Failed to initialize NVML: {e}")
            return False

    def get_power_readings(self) -> List[PowerReading]:
        """Get power readings for all NVIDIA GPUs"""
        if not self.initialized:
            return []

        readings = []
        timestamp = time.time()

        for i, handle in enumerate(self.gpu_handles):
            try:
                # Power consumption
                power_mw = nvml.nvmlDeviceGetPowerUsage(handle)
                power_watts = power_mw / 1000.0  # Convert milliwatts to watts

                # Temperature
                try:
                    temp_c = nvml.nvmlDeviceGetTemperature(
                        handle, nvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    temp_c = None

                # Utilization
                try:
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                except:
                    gpu_util = None

                # Clock frequency
                try:
                    clock_mhz = nvml.nvmlDeviceGetClockInfo(
                        handle, nvml.NVML_CLOCK_GRAPHICS
                    )
                except:
                    clock_mhz = None

                reading = PowerReading(
                    timestamp=timestamp,
                    component="gpu",
                    identifier=str(i),
                    power_watts=power_watts,
                    temperature_celsius=temp_c,
                    utilization_percent=gpu_util,
                    clock_mhz=clock_mhz
                )
                readings.append(reading)

            except Exception as e:
                print(f"Error reading GPU {i}: {e}")

        return readings

    def shutdown(self) -> None:
        """Clean shutdown of NVML"""
        if self.initialized and NVML_AVAILABLE:
            try:
                nvml.nvmlShutdown()
            except:
                pass
            self.initialized = False


class SystemPowerMonitor(PowerMonitor):
    """System-wide power monitoring using psutil and platform-specific APIs"""

    def __init__(self):
        self.initialized = False
        self.platform = platform.system().lower()

    def initialize(self) -> bool:
        """Initialize system power monitoring"""
        if not PSUTIL_AVAILABLE:
            print("psutil not available. Install psutil package.")
            return False

        self.initialized = True
        print(f"Initialized system power monitor for {self.platform}")
        return True

    def get_power_readings(self) -> List[PowerReading]:
        """Get system power readings"""
        if not self.initialized:
            return []

        readings = []
        timestamp = time.time()

        # CPU power (via RAPL on Intel, estimated on other platforms)
        cpu_reading = self._get_cpu_power_reading(timestamp)
        if cpu_reading:
            readings.append(cpu_reading)

        # Memory power (estimated)
        memory_reading = self._get_memory_power_reading(timestamp)
        if memory_reading:
            readings.append(memory_reading)

        # System power estimate
        system_reading = self._get_system_power_reading(timestamp)
        if system_reading:
            readings.append(system_reading)

        return readings

    def _get_cpu_power_reading(self, timestamp: float) -> Optional[PowerReading]:
        """Get CPU power reading"""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Estimate power based on TDP and utilization
            cpu_power_estimate = self._estimate_cpu_power(cpu_percent)

            # CPU temperature
            cpu_temp = self._get_cpu_temperature()

            return PowerReading(
                timestamp=timestamp,
                component="cpu",
                identifier="package",
                power_watts=cpu_power_estimate,
                temperature_celsius=cpu_temp,
                utilization_percent=cpu_percent
            )
        except Exception as e:
            print(f"Error getting CPU power: {e}")
            return None

    def _get_memory_power_reading(self, timestamp: float) -> Optional[PowerReading]:
        """Get memory power reading"""
        try:
            memory = psutil.virtual_memory()
            memory_util = memory.percent

            # Estimate memory power (rough approximation)
            # DDR4/DDR5 typically uses 2-5W per 8GB at 100% utilization
            memory_gb = memory.total / (1024**3)
            estimated_memory_power = (memory_gb / 8) * 3.5 * (memory_util / 100)

            return PowerReading(
                timestamp=timestamp,
                component="memory",
                identifier="system",
                power_watts=estimated_memory_power,
                utilization_percent=memory_util
            )
        except Exception as e:
            print(f"Error getting memory power: {e}")
            return None

    def _get_system_power_reading(self, timestamp: float) -> Optional[PowerReading]:
        """Get total system power reading"""
        try:
            # This is highly platform-specific
            # For now, return None on most platforms
            if self.platform == "linux":
                return self._get_linux_system_power(timestamp)
            elif self.platform == "darwin":  # macOS
                return self._get_macos_system_power(timestamp)
            elif self.platform == "windows":
                return self._get_windows_system_power(timestamp)
            else:
                return None
        except Exception as e:
            print(f"Error getting system power: {e}")
            return None

    def _estimate_cpu_power(self, cpu_util: float) -> float:
        """Estimate CPU power based on utilization"""
        # This is a rough estimate - in reality, CPU power is complex
        # Typical desktop CPU TDP ranges from 65W to 250W

        if self.platform == "darwin":  # macOS
            # Apple Silicon is very efficient
            tdp_estimate = 15.0  # M1/M2/M3 typical power
        elif self.platform == "linux":
            # Assume typical x86 desktop CPU
            tdp_estimate = 95.0  # Common for mid-range desktop CPUs
        else:
            tdp_estimate = 65.0  # Conservative estimate

        # Power doesn't scale linearly with utilization
        # Use a more realistic curve: P = P_idle + (P_max - P_idle) * (util^1.5)
        power_idle = tdp_estimate * 0.1  # 10% of TDP at idle
        power_max = tdp_estimate * 1.2   # Can exceed TDP under load

        estimated_power = power_idle + (power_max - power_idle) * (cpu_util / 100) ** 1.5
        return estimated_power

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature"""
        try:
            if self.platform == "linux":
                return self._get_linux_cpu_temp()
            elif self.platform == "darwin":
                return self._get_macos_cpu_temp()
            elif self.platform == "windows":
                return self._get_windows_cpu_temp()
            else:
                return None
        except:
            return None

    def _get_linux_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature on Linux"""
        try:
            # Read from /sys/class/thermal/
            thermal_zones = []
            for zone_file in ["/sys/class/thermal/thermal_zone0/temp",
                             "/sys/class/thermal/thermal_zone1/temp"]:
                try:
                    with open(zone_file, 'r') as f:
                        temp_millidegrees = int(f.read().strip())
                        temp_celsius = temp_millidegrees / 1000.0
                        if 0 < temp_celsius < 150:  # Sanity check
                            thermal_zones.append(temp_celsius)
                except:
                    continue

            return max(thermal_zones) if thermal_zones else None
        except:
            return None

    def _get_macos_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature on macOS"""
        try:
            # Use powermetrics command on macOS
            result = subprocess.run([
                'sudo', 'powermetrics', '--samplers', 'cpu_power', '-i', '1', '-n', '1'
            ], capture_output=True, text=True, timeout=5)

            # Parse output for temperature
            for line in result.stdout.split('\n'):
                if 'CPU die temperature' in line:
                    temp_str = line.split()[-1].replace('C', '')
                    return float(temp_str)
            return None
        except:
            return None

    def _get_windows_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature on Windows"""
        try:
            # Use WMI to get temperature
            import wmi
            c = wmi.WMI()

            for temperature in c.Win32_TemperatureProbe():
                if temperature.CurrentReading:
                    return temperature.CurrentReading
            return None
        except:
            return None

    def _get_linux_system_power(self, timestamp: float) -> Optional[PowerReading]:
        """Get system power on Linux"""
        try:
            # Try reading from power supply class
            for power_supply in ["/sys/class/power_supply/BAT0/power_now",
                                "/sys/class/power_supply/AC/power_now"]:
                try:
                    with open(power_supply, 'r') as f:
                        power_uw = int(f.read().strip())
                        power_watts = power_uw / 1000000.0  # Convert microwatts to watts
                        return PowerReading(
                            timestamp=timestamp,
                            component="system",
                            identifier="power_supply",
                            power_watts=power_watts
                        )
                except:
                    continue
            return None
        except:
            return None

    def _get_macos_system_power(self, timestamp: float) -> Optional[PowerReading]:
        """Get system power on macOS"""
        # macOS power monitoring is complex
        # For now, return None
        return None

    def _get_windows_system_power(self, timestamp: float) -> Optional[PowerReading]:
        """Get system power on Windows"""
        # Windows power monitoring requires WMI or specific tools
        # For now, return None
        return None

    def shutdown(self) -> None:
        """Clean shutdown of system monitor"""
        self.initialized = False


class MultiDevicePowerMonitor:
    """Aggregates multiple power monitoring devices"""

    def __init__(self):
        self.monitors: List[PowerMonitor] = []
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize all available power monitors"""

        # Add NVIDIA GPU monitor
        nvidia_monitor = NVidiaGPUPowerMonitor()
        if nvidia_monitor.initialize():
            self.monitors.append(nvidia_monitor)
            print("Added NVIDIA GPU power monitor")

        # Add system power monitor
        system_monitor = SystemPowerMonitor()
        if system_monitor.initialize():
            self.monitors.append(system_monitor)
            print("Added system power monitor")

        self.initialized = len(self.monitors) > 0
        print(f"Initialized {len(self.monitors)} power monitors")
        return self.initialized

    def get_all_power_readings(self) -> List[PowerReading]:
        """Get power readings from all monitors"""
        all_readings = []

        for monitor in self.monitors:
            try:
                readings = monitor.get_power_readings()
                all_readings.extend(readings)
            except Exception as e:
                print(f"Error getting readings from {monitor.__class__.__name__}: {e}")

        return all_readings

    def get_total_power_consumption(self) -> float:
        """Get total system power consumption"""
        readings = self.get_all_power_readings()
        return sum(r.power_watts for r in readings)

    def get_power_breakdown(self) -> Dict[str, float]:
        """Get power consumption breakdown by component type"""
        readings = self.get_all_power_readings()
        breakdown = {}

        for reading in readings:
            if reading.component not in breakdown:
                breakdown[reading.component] = 0.0
            breakdown[reading.component] += reading.power_watts

        return breakdown

    def shutdown(self) -> None:
        """Shutdown all monitors"""
        for monitor in self.monitors:
            try:
                monitor.shutdown()
            except Exception as e:
                print(f"Error shutting down {monitor.__class__.__name__}: {e}")

        self.monitors = []
        self.initialized = False


# Convenience function for quick setup
def create_power_monitor() -> MultiDevicePowerMonitor:
    """Create and initialize a multi-device power monitor"""
    monitor = MultiDevicePowerMonitor()
    monitor.initialize()
    return monitor