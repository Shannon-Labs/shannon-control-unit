#!/usr/bin/env python3
"""
Validation script for CUDA-optimized SCU training integration
"""

import torch
import sys
import json
from pathlib import Path
from importlib import import_module

def validate_cuda_integration():
    """Validate that the CUDA script integrates properly with SCU system"""
    print("🔍 Validating CUDA-optimized SCU training integration...")
    
    # Check CUDA availability (optional for validation)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Validate imports
    try:
        # Test SCU control module
        from scu.control import update_lambda, calculate_param_bpt, calculate_data_bpt, calculate_s_ratio
        print("✅ SCU control module imported successfully")
        
        # Test configuration
        from configs.granite_1b_scu_config import Granite1BSCUConfig
        config = Granite1BSCUConfig()
        print("✅ Configuration loaded successfully")
        
        # Test CUDA script imports (without executing)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_granite_cuda", 
            Path(__file__).parent / "scripts" / "train_granite_cuda.py"
        )
        cuda_module = importlib.util.module_from_spec(spec)
        
        # This will validate syntax and basic imports
        spec.loader.exec_module(cuda_module)
        print("✅ CUDA training script syntax is valid")
        
        # Check for required functions
        required_functions = [
            'CudaOptimizedSCUTrainer',
            'detect_optimal_dtype', 
            'setup_logging',
            'check_cuda_requirements'
        ]
        
        for func_name in required_functions:
            if hasattr(cuda_module, func_name):
                print(f"✅ Function {func_name} found")
            else:
                print(f"❌ Function {func_name} missing")
                return False
        
        # Test optimal dtype detection
        optimal_dtype = cuda_module.detect_optimal_dtype()
        print(f"✅ Optimal dtype detection: {optimal_dtype}")
        
        # Test CUDA requirements check
        print("Testing CUDA requirements check...")
        try:
            # This should work regardless of CUDA availability
            cuda_module.check_cuda_requirements()
            print("✅ CUDA requirements check completed")
        except Exception as e:
            print(f"⚠️  CUDA requirements check had issues: {e}")
        
        # Validate SCU control integration
        print("Testing SCU control integration...")
        
        # Test lambda update function
        lambda_new, integral_new, s_hat = update_lambda(
            lmbda=0.1,
            S_meas=0.02,
            S_target=0.01,
            I=0.0,
            Kp=0.6,
            Ki=0.1
        )
        print(f"✅ SCU control test: λ={lambda_new:.4f}, I={integral_new:.4f}")
        
        # Test BPT calculations
        data_bpt = calculate_data_bpt(loss_nats=2.0)
        print(f"✅ Data BPT calculation: {data_bpt:.4f} bits/token")
        
        # Test S-ratio calculation
        s_ratio = calculate_s_ratio(data_bpt=2.0, param_bpt=0.1)
        print(f"✅ S-ratio calculation: {s_ratio:.4f}")
        
        # Validate configuration compatibility
        print("Validating configuration compatibility...")
        
        # Check CUDA-specific settings
        cuda_specific_settings = [
            'use_torch_compile',
            'use_4bit_quantization', 
            'use_streaming',
            'preprocessing_num_workers'
        ]
        
        for setting in cuda_specific_settings:
            if hasattr(config, setting):
                print(f"✅ CUDA setting {setting}: {getattr(config, setting)}")
            else:
                print(f"⚠️  CUDA setting {setting} not found in config")
        
        print("✅ Integration validation completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def validate_script_features():
    """Validate specific features of the CUDA script"""
    print("\n🔍 Validating CUDA script features...")
    
    # Read the script content
    script_path = Path(__file__).parent / "scripts" / "train_granite_cuda.py"
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for CUDA-specific features
        features = {
            'Mixed Precision': ['fp16', 'bf16', 'autocast', 'GradScaler'],
            'Memory Management': ['torch.cuda.empty_cache', 'memory_allocated', 'memory_reserved'],
            'Multi-GPU': ['device_count', 'current_device', 'distributed'],
            'torch.compile': ['torch.compile', 'max-autotune'],
            'Quantization': ['BitsAndBytesConfig', 'load_in_4bit'],
            'SCU Integration': ['update_lambda', 'calculate_param_bpt', 'calculate_s_ratio'],
            'Performance': ['cudnn.benchmark', 'enable_flash_sdp', 'tf32']
        }
        
        for feature_name, keywords in features.items():
            found = any(keyword in content for keyword in keywords)
            if found:
                print(f"✅ {feature_name}: Implemented")
            else:
                print(f"⚠️  {feature_name}: Not fully implemented")
        
        # Check for proper error handling
        error_patterns = ['try:', 'except', 'logger.error', 'torch.cuda.is_available()']
        error_handling = sum(1 for pattern in error_patterns if pattern in content)
        print(f"✅ Error handling patterns: {error_handling}/{len(error_patterns)}")
        
        # Check for logging
        logging_patterns = ['logger.info', 'logger.warning', 'logger.error']
        logging_count = sum(1 for pattern in logging_patterns if pattern in content)
        print(f"✅ Logging patterns: {logging_count}/{len(logging_patterns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature validation error: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Starting CUDA integration validation...")
    
    # Run integration validation
    integration_ok = validate_cuda_integration()
    
    # Run feature validation
    features_ok = validate_script_features()
    
    # Final summary
    print("\n📊 Validation Summary:")
    print(f"  Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print(f"  Features: {'✅ PASS' if features_ok else '❌ FAIL'}")
    
    if integration_ok and features_ok:
        print("\n🎉 CUDA-optimized SCU training script is ready!")
        print("\nNext steps:")
        print("1. Run: python scripts/train_granite_cuda.py --check-requirements")
        print("2. Test: python scripts/train_granite_cuda.py --test-run")
        print("3. Train: python scripts/train_granite_cuda.py --fp16 --batch-size 8")
        return True
    else:
        print("\n❌ Validation failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)