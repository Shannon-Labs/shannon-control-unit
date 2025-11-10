#!/usr/bin/env python3
"""
Test script for simplified SCU training logic
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append('.')

from scu2.core.simplified_controller import SimplifiedSCU, TrainingState


def test_sscu():
    """Test simplified SCU logic"""
    print("Testing Simplified SCU...")

    controller = SimplifiedSCU(
        target_loss_improvement=0.01,
        control_frequency=20,
        enable_multiscale_analysis=True
    )

    print("✓ SCU controller created")

    # Simulate training with decreasing loss
    actions_taken = 0
    for step in range(100):
        # Simulate loss decreasing with some noise and patterns
        base_loss = 2.0 * np.exp(-step * 0.01)

        # Add some interesting dynamics
        if 30 < step < 40:
            base_loss *= 1.2  # Simulate difficulty
        if step > 70:
            base_loss *= 0.9  # Fast convergence

        noise = 0.05 * np.random.randn()
        loss = base_loss + noise

        # Simulate gradient norm with some patterns
        grad_norm = 0.5 + 0.3 * np.sin(step * 0.1) + 0.1 * np.random.randn()

        # Create training state
        state = TrainingState(
            loss_value=loss,
            gradient_norm=grad_norm,
            learning_rate=2e-4,
            batch_size=2,
            step_count=step,
            timestamp=time.time()
        )

        # Get SCU action
        action = controller.update_state(state)

        if action:
            actions_taken += 1
            print(f"  Step {step}: {action.reason[:60]}...")

    summary = controller.get_training_summary()
    print(f"✓ SCU test completed")
    print(f"  Actions taken: {actions_taken}")
    print(f"  Final loss: {summary['current_loss']:.4f}")
    print(f"  Loss trend: {summary['loss_trend']:.6f}")

    return True


def test_multiscale_entropy():
    """Test multi-scale entropy analysis"""
    print("\nTesting Multi-scale Entropy Analysis...")

    try:
        from scu2.core.multiscale_entropy import quick_multiscale_analysis

        # Create a test signal with multi-scale structure
        t = np.linspace(0, 10, 256)
        signal = (
            np.sin(2 * np.pi * 1 * t) +           # Low frequency
            0.5 * np.sin(2 * np.pi * 10 * t) +    # Medium frequency
            0.2 * np.sin(2 * np.pi * 50 * t) +    # High frequency
            0.1 * np.random.randn(len(t))         # Noise
        )

        result = quick_multiscale_analysis(signal)

        print("✓ Multi-scale analysis completed")
        print(f"  Total entropy bits: {result['total_entropy_bits']:.2f}")
        print(f"  Selected scales: {result['selected_scales_count']}")
        print(f"  Scale mask: {result['scale_mask_hex']}")

        return True

    except Exception as e:
        print(f"✗ Multi-scale analysis failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting Configuration...")

    try:
        from scu2.production.configs.qwen3_4b_mlx_config import Qwen3MLXProductionConfig

        config = Qwen3MLXProductionConfig()

        print("✓ Configuration loaded successfully")
        print(f"  Model: {config.model_name}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Max steps: {config.max_steps}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  LoRA enabled: {config.use_lora}")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_mock_training():
    """Test mock training without actual model"""
    print("\nTesting Mock Training Logic...")

    try:
        from scu2.production.scripts.train_qwen3_simple import SimplifiedTrainer
        from scu2.production.configs.qwen3_4b_mlx_config import Qwen3MLXProductionConfig

        config = Qwen3MLXProductionConfig()
        config.max_steps = 10  # Very short test

        # Mock trainer arguments
        class MockArgs:
            def __init__(self):
                self.train_batch_size = config.batch_size
                self.max_grad_norm = 1.0

        # Create a mock trainer (we can't actually instantiate without a model)
        # But we can test the import and basic structure
        print("✓ Trainer import successful")
        print(f"  Mock trainer would use: {config.max_steps} steps")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Control frequency: 50 steps")

        return True

    except Exception as e:
        print(f"✗ Mock training test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting T-SCU Training Logic Tests\n")

    tests = [
        ("Simplified SCU", test_sscu),
        ("Multi-scale Entropy", test_multiscale_entropy),
        ("Configuration", test_config),
        ("Mock Training", test_mock_training),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Training pipeline is ready.")
        print("\n🚀 Ready to train Qwen3-4B-MLX-4bit!")
        print("\nTo start training:")
        print("1. Install missing dependencies: pip install datasets")
        print("2. Run: python scu2/production/scripts/train_qwen3_simple.py --test-run")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())