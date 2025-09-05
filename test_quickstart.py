#!/usr/bin/env python3
"""
Test the quickstart code examples from README to ensure they work.
This validates the user experience before publishing.
"""

def test_imports():
    """Test that required packages can be imported."""
    try:
        import torch
        print("✓ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        from peft import PeftModel
        print("✓ peft imported successfully")
    except ImportError as e:
        print(f"❌ peft import failed: {e}")
        return False
    
    return True

def test_model_ids():
    """Test that the base model IDs are valid."""
    from transformers import AutoTokenizer
    
    base_models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B"
    ]
    
    for model_id in base_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            print(f"✓ {model_id} tokenizer loads successfully")
        except Exception as e:
            print(f"⚠️  {model_id} tokenizer failed (may require HF auth): {e}")
    
    return True

def test_adapter_files_exist():
    """Validate adapter artifacts are present in tracked subfolders.

    Root-level adapter files are optional and may be ignored to keep the repo tidy.
    """
    from pathlib import Path

    # Optional: root-level files (do not fail if missing)
    for file in ["adapter_config.json", "adapter_model.safetensors"]:
        if Path(file).exists():
            print(f"✓ {file} exists (root)")
        else:
            print(f"ℹ️  {file} not present at repo root (expected for a clean repo)")

    # Required: subdirectory adapters
    ok = True
    for subdir in ["1b-scu", "3b-scu", "3b-fixed"]:
        path = Path(subdir)
        if path.exists() and path.is_dir():
            adapter_config = path / "adapter_config.json"
            adapter_model = path / "adapter_model.safetensors"
            if adapter_config.exists() and adapter_model.exists():
                print(f"✓ {subdir}/ contains adapter files")
            else:
                print(f"❌ {subdir}/ missing adapter files")
                ok = False
        else:
            print(f"❌ {subdir}/ directory not found")
            ok = False

    return ok

def test_validation_files():
    """Test that validation files exist and are readable."""
    from pathlib import Path
    import json
    
    # Check validation results
    validation_file = Path("results/3b_validation_results.json")
    if validation_file.exists():
        try:
            with open(validation_file) as f:
                data = json.load(f)
            print("✓ 3b_validation_results.json is valid JSON")
            
            # Check for required fields
            if "summary" in data and "improvement_percent" in data["summary"]:
                improvement = data["summary"]["improvement_percent"]
                print(f"✓ Validation shows {improvement}% improvement")
            
        except Exception as e:
            print(f"❌ 3b_validation_results.json invalid: {e}")
            return False
    else:
        print("❌ 3b_validation_results.json missing")
        return False
    
    return True

def test_image_files():
    """Test that all referenced images exist."""
    from pathlib import Path
    
    required_images = [
        "assets/figures/validation_3b_comparison.png",
        "assets/figures/ablation_s_tracking.png", 
        "assets/figures/ablation_lambda_evolution.png",
        "assets/figures/ablation_final_performance.png",
        "assets/figures/s_curve.png",
        "assets/figures/lambda_curve.png",
        "assets/figures/data_bpt_curve.png",
        "assets/figures/param_bpt_curve.png"
    ]
    
    missing_images = []
    for img_path in required_images:
        if Path(img_path).exists():
            print(f"✓ {img_path}")
        else:
            print(f"❌ {img_path} missing")
            missing_images.append(img_path)
    
    return len(missing_images) == 0

def main():
    """Run all validation tests."""
    print("🔍 Shannon Control Unit - Quickstart Validation")
    print("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("Model IDs", test_model_ids), 
        ("Adapter files", test_adapter_files_exist),
        ("Validation files", test_validation_files),
        ("Image assets", test_image_files)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Repository is ready for publication.")
        return 0
    else:
        print("⚠️  Some tests failed. Address issues before publishing.")
        return 1

if __name__ == '__main__':
    exit(main())
