#!/usr/bin/env python3
"""
Inspect OLMO model architecture to understand LoRA target modules
"""
import torch
from transformers import AutoModelForCausalLM

def inspect_olmo_model():
    print("🔍 Inspecting OLMO Model Architecture")
    print("=" * 50)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained('allenai/OLMo-7B', trust_remote_code=True)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Model config: {model.config}")
    
    # Inspect layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"\n📊 Model Structure:")
        print(f"  Base model type: {type(model.model).__name__}")
        print(f"  Number of layers: {len(model.model.layers)}")
        
        # Inspect first layer
        first_layer = model.model.layers[0]
        print(f"\n🔬 First Layer Analysis:")
        print(f"  Layer type: {type(first_layer).__name__}")
        
        # List all modules in the first layer
        print(f"\n📝 Module Names in First Layer:")
        for name, module in first_layer.named_modules():
            if any(keyword in name.lower() for keyword in ['proj', 'attention', 'mlp', 'ffn']):
                print(f"  {name}: {type(module).__name__}")
    
    # Also check the model directly
    print(f"\n🎯 Direct Model Modules (first 20):")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 20:  # Limit output
            break
        if any(keyword in name.lower() for keyword in ['proj', 'attention', 'mlp', 'ffn', 'q', 'k', 'v', 'o', 'gate', 'up', 'down']):
            print(f"  {name}: {type(module).__name__}")
    
    # Look for specific attention and MLP modules
    print(f"\n🔍 Searching for attention components:")
    attention_modules = []
    mlp_modules = []
    
    for name, module in model.named_modules():
        if 'attention' in name.lower() and len(name.split('.')) >= 3:  # Deeper modules
            attention_modules.append(name)
        if any(keyword in name.lower() for keyword in ['mlp', 'ffn']) and len(name.split('.')) >= 3:
            mlp_modules.append(name)
    
    print(f"Attention modules found: {attention_modules[:5]}...")  # Show first 5
    print(f"MLP modules found: {mlp_modules[:5]}...")  # Show first 5

if __name__ == "__main__":
    inspect_olmo_model()
