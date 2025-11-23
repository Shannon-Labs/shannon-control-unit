#!/usr/bin/env python3
"""
Check if OLMO model is ready for training
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_olmo_model():
    print("🧪 Checking OLMO Model Status")
    print("=" * 40)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check model availability
    print("\n📥 Checking model download status...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/OLMo-7B', 
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("✅ OLMO model downloaded successfully!")
        
        # Check tokenizer
        tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-7B')
        print("✅ OLMO tokenizer loaded successfully!")
        
        # Model info
        print(f"\n📊 Model Information:")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Model type: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    check_olmo_model()
