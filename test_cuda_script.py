#!/usr/bin/env python3
"""
Test script for CUDA-optimized SCU training script
"""

import subprocess
import sys
import torch
from pathlib import Path

def test_cuda_script():
    """Test the CUDA training script with minimal configuration"""
    print("🔍 Testing CUDA-optimized SCU training script...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Skipping CUDA tests.")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"📊 CUDA devices: {torch.cuda.device_count()}")
    
    # Test script import
    try:
        sys.path.append(str(Path(__file__).parent))
        from scripts.train_granite_cuda import (
            CudaOptimizedSCUTrainer, 
            detect_optimal_dtype,
            create_model_and_tokenizer_cuda,
            load_and_prepare_dataset_cuda
        )
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test dtype detection
    optimal_dtype = detect_optimal_dtype()
    print(f"✅ Detected optimal dtype: {optimal_dtype}")
    
    # Test script help
    try:
        result = subprocess.run([
            sys.executable, "scripts/train_granite_cuda.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Help command works")
            # Check for CUDA-specific arguments
            if "--fp16" in result.stdout and "--multi-gpu" in result.stdout:
                print("✅ CUDA-specific arguments present")
            else:
                print("⚠️  Some CUDA arguments missing")
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Help command timed out")
        return False
    except Exception as e:
        print(f"❌ Help command error: {e}")
        return False
    
    print("✅ CUDA training script validation completed successfully!")
    return True

if __name__ == "__main__":
    success = test_cuda_script()
    sys.exit(0 if success else 1)