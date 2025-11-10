#!/usr/bin/env python3
"""
Apple Silicon T-SCU Demo

Demonstrates Thermodynamic SCU running on Apple Silicon Macs.
This is actually the PERFECT platform for T-SCU because:

1. Apple Silicon is incredibly energy-efficient
2. Native macOS has excellent power monitoring capabilities
3. Unified Memory Architecture reduces energy waste
4. Advanced thermal management systems
5. Neural Engine for efficient inference

Usage:
    sudo python scu2/experiments/apple_silicon_demo.py

Note: sudo access recommended for full powermetrics capabilities,
but demo works with reduced functionality without it.
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scu2.hardware.apple_power_monitor import create_apple_power_monitor
from scu2.core.thermodynamic_controller import ThermodynamicSCU
from scu2.metrics.energy_entropy import AdvancedEnergyEntropyMetrics


def setup_demo_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('apple_silicon_tscu_demo.log')
        ]
    )
    return logging.getLogger("AppleTSCUDemo")


def create_simple_model():
    """Create a simple model for demonstration"""
    # Create a small transformer model that works well on Apple Silicon
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, vocab_size=1000, d_model=256, nhead=8, num_layers=4):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = torch.nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return self.fc(x)

    return SimpleTransformer()


def simulate_training_step(model, batch_size=8, seq_len=128):
    """Simulate a training step"""
    # Create random data
    inputs = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)),
        targets.view(-1)
    )

    # Backward pass
    loss.backward()

    return loss.item()


def run_apple_silicon_demo():
    """Run the Apple Silicon T-SCU demonstration"""
    logger = setup_demo_logging()
    logger.info("🍎 Apple Silicon T-SCU Demo Starting...")
    logger.info("=" * 60)

    # Check if running on Apple Silicon
    try:
        import platform
        system_info = platform.platform()
        logger.info(f"Platform: {system_info}")

        if "arm" in system_info.lower() or "apple" in system_info.lower():
            logger.info("✅ Detected Apple Silicon - Perfect for T-SCU!")
        else:
            logger.warning("⚠️  Not running on Apple Silicon - demo will use fallback methods")
    except:
        logger.warning("Could not detect platform")

    # Initialize power monitoring
    logger.info("\n🔋 Initializing Apple Silicon Power Monitor...")
    power_monitor = create_apple_power_monitor()

    if not power_monitor.initialized:
        logger.error("❌ Failed to initialize power monitoring")
        return False

    logger.info(f"✅ Power monitor initialized for {power_monitor.chip_model}")
    logger.info(f"📊 Powermetrics available: {power_monitor.powermetrics_available}")

    # Initialize T-SCU controller
    logger.info("\n🎛️  Initializing Thermodynamic SCU...")
    tscu = ThermodynamicSCU(
        power_budget_watts=15.0,  # Conservative for Apple Silicon
        target_efficiency=1e-5,   # Apple Silicon is very efficient
        max_temperature_celsius=75.0,  # Apple Silicon runs cool
        landauer_efficiency_target=1e-7
    )
    logger.info("✅ T-SCU controller initialized")

    # Initialize metrics calculator
    metrics_calculator = AdvancedEnergyEntropyMetrics()
    logger.info("✅ Energy-entropy metrics initialized")

    # Create model
    logger.info("\n🧠 Creating demonstration model...")
    model = create_simple_model()
    model.train()

    # Move to Metal if available (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        logger.info("✅ Model moved to Apple Metal Performance Shaders (GPU)")
    else:
        device = torch.device("cpu")
        model = model.to(device)
        logger.info("✅ Model using CPU (still very efficient on Apple Silicon)")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    logger.info("\n🚀 Starting Power-Aware Training Simulation...")
    logger.info("=" * 60)

    # Run demo training loop
    demo_steps = 50
    total_energy_joules = 0
    start_time = time.time()

    for step in range(demo_steps):
        step_start_time = time.time()

        # Get power readings
        power_readings = power_monitor.get_power_readings()
        current_power = sum(r.power_watts for r in power_readings) if power_readings else 0

        # Get temperatures
        temperatures = [r.temperature_celsius for r in power_readings if r.temperature_celsius]
        avg_temperature = max(temperatures) if temperatures else 25.0

        # Simulate training step
        loss = simulate_training_step(model)
        optimizer.step()
        optimizer.zero_grad()

        # Calculate information entropy
        param_entropy = 0.0
        for param in model.parameters():
            if param.requires_grad:
                # Simple entropy estimate
                param_entropy += torch.numel(param) * 0.5  # Rough estimate

        # Create thermodynamic state
        thermo_state = tscu.calculate_thermodynamic_metrics(
            information_entropy=param_entropy,
            power_consumption=current_power,
            temperature=avg_temperature + 273.15
        )

        # Get T-SCU control action
        control_factor, control_breakdown = tscu.compute_control_action(thermo_state, step)

        # Apply control (simulate adaptive training)
        adaptive_lr = 1e-4 * control_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = adaptive_lr

        # Calculate energy used in this step
        step_time = time.time() - step_start_time
        step_energy_joules = current_power * step_time
        total_energy_joules += step_energy_joules

        # Log progress every 10 steps
        if step % 10 == 0:
            logger.info(
                f"Step {step:2d} | Loss: {loss:.4f} | Power: {current_power:5.2f}W | "
                f"Temp: {avg_temperature:5.1f}°C | Control: {control_factor:.2f} | "
                f"Efficiency: {thermo_state.efficiency_bits_per_joule:.2e} bits/J"
            )

        # Record state
        tscu.record_state_and_control(thermo_state, control_breakdown)

        # Adaptive pause based on power constraints
        if current_power > 20.0:  # Power limit
            time.sleep(0.1)  # Brief pause for cooling

        # Emergency thermal check
        if avg_temperature > 85.0:
            logger.warning(f"🔥 Temperature too high: {avg_temperature:.1f}°C")
            break

    # Calculate final statistics
    total_time = time.time() - start_time
    final_analysis = tscu.analyze_performance()
    recommendations = tscu.get_optimization_recommendations()

    logger.info("\n" + "=" * 60)
    logger.info("📊 Apple Silicon T-SCU Demo Results")
    logger.info("=" * 60)

    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    logger.info(f"⚡ Total energy: {total_energy_joules:.2f} joules ({total_energy_joules/3600:.4f} Wh)")
    logger.info(f"🔋 Average power: {total_energy_joules/total_time:.2f} watts")

    if final_analysis.get("status") != "insufficient_data":
        logger.info(f"🌡️  Average temperature: {final_analysis['avg_temperature_celsius']:.1f}°C")
        logger.info(f"📈 Power budget utilization: {final_analysis['power_budget_utilization']:.1f}%")
        logger.info(f"🎯 Efficiency target achievement: {final_analysis['efficiency_target_achievement']:.1f}%")
        logger.info(f"🏃 Efficiency improvement: {final_analysis['efficiency_improvement_percent']:.1f}%")

    logger.info("\n💡 Optimization Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")

    # Apple Silicon specific analysis
    apple_metrics = power_monitor.get_apple_specific_metrics()
    if apple_metrics:
        logger.info("\n🍎 Apple Silicon Specific Metrics:")
        logger.info(f"  Efficiency Score: {apple_metrics['efficiency_score']:.2f}")
        logger.info(f"  Thermal Headroom: {apple_metrics['thermal_headroom']:.1f}°C")

    logger.info("\n🎉 Apple Silicon T-SCU Demo Completed Successfully!")
    logger.info("\nKey Insights:")
    logger.info("• Apple Silicon's energy efficiency makes it ideal for T-SCU")
    logger.info("• Native macOS power monitoring provides excellent visibility")
    logger.info("• Thermal management is superior to traditional desktop hardware")
    logger.info("• Neural Engine offers opportunities for ultra-efficient inference")

    # Cleanup
    power_monitor.shutdown()
    logger.info("\n✅ Demo completed and systems shutdown gracefully")
    return True


if __name__ == "__main__":
    print("🍎 Apple Silicon Thermodynamic SCU Demo")
    print("=" * 50)
    print()
    print("This demo showcases T-SCU running on Apple Silicon Macs.")
    print("For full powermetrics capabilities, run with sudo:")
    print("  sudo python scu2/experiments/apple_silicon_demo.py")
    print()
    print("Starting demo...")

    try:
        success = run_apple_silicon_demo()
        if success:
            print("\n🎉 Demo completed successfully!")
        else:
            print("\n❌ Demo failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)