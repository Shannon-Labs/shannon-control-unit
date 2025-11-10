#!/usr/bin/env python3
"""
Quick test of Ultra-Active SCU with Granite-4.0-Micro

This script performs a minimal test to ensure the ultra-active SCU system
works correctly before starting full training.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_ultra_scu():
    """Test ultra-active SCU controller"""
    print("🧪 Testing Ultra-Active SCU Controller...")

    try:
        from scu2.core.ultra_active_controller import UltraActiveSCU, UltraTrainingState
        import time
        import numpy as np

        # Create ultra-active SCU
        controller = UltraActiveSCU(
            target_s_ratio=0.008,
            kp=1.2,
            ki=0.25,
            enable_adaptive_gains=True,
            enable_predictive_control=True,
            enable_thermal_aware=True
        )

        print("✅ Ultra-Active SCU initialized successfully")

        # Test with simulated data
        for step in range(10):
            # Simulate challenging S ratio dynamics
            s_ratio = 0.008 + 0.002 * np.sin(step * 0.5) + np.random.randn() * 0.0005

            state = UltraTrainingState(
                loss_value=2.5 + np.random.randn() * 0.1,
                data_bpt=3.6 + np.random.randn() * 0.1,
                param_bpt=s_ratio * 3.6,
                s_ratio=s_ratio,
                gradient_norm=0.3 + np.random.randn() * 0.05,
                learning_rate=1e-4,
                lambda_value=controller.lambda_current,
                step_count=step,
                timestamp=time.time(),
                temperature=70.0 + np.random.randn() * 2.0
            )

            action = controller.update_state(state)

            print(f"Step {step}: S={s_ratio:.4f}, λ={action.new_lambda:.4f}, Effort={action.control_effort:.4f}")
            print(f"  Reason: {action.reason}")

        # Get summary
        summary = controller.get_comprehensive_summary()
        print(f"\n📊 Test Summary:")
        print(f"  Control frequency: {summary['control_frequency_percent']:.1f}%")
        print(f"  Current S ratio: {summary['current_s_ratio']:.4f}")
        print(f"  Current lambda: {summary['current_lambda']:.4f}")
        print(f"  Control actions: {summary['control_actions_applied']}")

        print("✅ Ultra-Active SCU test passed!")
        return True

    except Exception as e:
        print(f"❌ Ultra-Active SCU test failed: {e}")
        return False

def test_granite_loading():
    """Test Granite model loading"""
    print("\n🧪 Testing Granite-4.0-Micro loading...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Test loading
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-micro")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-4.0-micro",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Max length: {tokenizer.model_max_length}")

        # Test tokenization
        test_text = "Hello, world! This is a test."
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test passed: {len(inputs['input_ids'][0])} tokens")

        return True

    except Exception as e:
        print(f"❌ Granite loading test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")

    try:
        from scu2.production.configs.granite_micro_ultra_config import GraniteMicroUltraSCUConfig

        config = GraniteMicroUltraSCUConfig()

        print(f"✅ Configuration loaded successfully!")
        print(f"   Model: {config.model_name}")
        print(f"   Target S ratio: {config.target_s_ratio:.3%}")
        print(f"   Lambda range: {config.lambda_min} to {config.lambda_max}")
        print(f"   PI gains: Kp={config.kp}, Ki={config.ki}")
        print(f"   Control frequency: Every step")
        print(f"   Max steps: {config.max_steps}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Ultra-Active SCU + Granite Test Suite")
    print("=" * 50)

    tests = [
        test_configuration,
        test_ultra_scu,
        test_granite_loading,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for ultra-active training!")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)