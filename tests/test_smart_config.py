from scu_api.service.smart_config import auto_configure, detect_model_params


def test_model_detection():
    assert detect_model_params("gpt2")[0] == 0.124
    assert detect_model_params("Llama-3.2-3B")[0] == 3.0


def test_auto_configure_defaults():
    config = auto_configure("gpt2")
    assert config.target_s == 0.005
    assert config.lora_r == 8
