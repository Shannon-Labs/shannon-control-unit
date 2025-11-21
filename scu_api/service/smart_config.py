import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from scu_api.config import TrainingConfig

# Parameter counts in billions
MODEL_PARAMS = {
    # GPT
    "gpt2": 0.124,
    "distilgpt2": 0.082,
    "microsoft/DialoGPT-small": 0.124,
    "microsoft/DialoGPT-medium": 0.355,
    "microsoft/DialoGPT-large": 0.774,

    # Llama
    "Llama-3.2-1B": 1.0,
    "Llama-3.2-3B": 3.0,
    "Llama-3.1-8B": 8.0,
    "Llama-3.1-70B": 70.0,

    # EleutherAI
    "EleutherAI/gpt-neo-125m": 0.125,
    "EleutherAI/gpt-neo-1.3B": 1.3,
    "EleutherAI/pythia-70m": 0.07,
    "EleutherAI/pythia-1b": 1.0,
    "EleutherAI/pythia-2.8b": 2.8,

    # Phi
    "microsoft/Phi-3-mini-4k-instruct": 3.8,
    "microsoft/Phi-3-small-8k-instruct": 7.0,

    # Gemma
    "google/gemma-2b": 2.0,
    "google/gemma-7b": 7.0,

    # BlenderBot
    "facebook/blenderbot-1B-distill": 1.0,
    "facebook/blenderbot-3B": 3.0,

    # Falcon
    "tiiuae/falcon-7b": 7.0,
    "tiiuae/falcon-40b": 40.0,
}

CONFIG_SCALES = {
    (0.0, 0.2): {"target_s": 0.005, "lora_r": 8, "lora_alpha": 16, "batch_size": 8, "lr": 1e-4, "block_size": 512},
    (0.2, 1.0): {"target_s": 0.008, "lora_r": 16, "lora_alpha": 32, "batch_size": 4, "lr": 8e-5, "block_size": 1024},
    (1.0, 4.0): {"target_s": 0.01, "lora_r": 16, "lora_alpha": 32, "batch_size": 2, "lr": 5e-5, "block_size": 2048},
    (4.0, 10.0): {"target_s": 0.02, "lora_r": 32, "lora_alpha": 64, "batch_size": 1, "lr": 3e-5, "block_size": 2048},
    (10.0, 100.0): {"target_s": 0.03, "lora_r": 64, "lora_alpha": 128, "batch_size": 1, "lr": 2e-5, "block_size": 4096},
}


def detect_model_params(model_id: str) -> Tuple[float, Dict]:
    """Detect model size in billions and return config."""
    model_id_lower = model_id.lower()

    for known_id, params in MODEL_PARAMS.items():
        if known_id.lower() in model_id_lower:
            return params, _get_config_for_params(params)

    patterns = [
        (r"[-_](\d+)(?:b|B)(?:[-_]|$)", 1.0),
        (r"[-_](\d{2,3})(?:b|B)(?:[-_]|$)", 1.0),
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, model_id)
        if match:
            param_count = int(match.group(1)) * multiplier
            return param_count, _get_config_for_params(param_count)

    return 1.0, CONFIG_SCALES[(1.0, 4.0)]


def _get_config_for_params(param_count: float) -> Dict:
    for (min_p, max_p), config in CONFIG_SCALES.items():
        if min_p <= param_count < max_p:
            return config.copy()
    return CONFIG_SCALES[(10.0, 100.0)].copy()


def auto_configure(
    model_id: str,
    train_data: Optional[str] = None,
    adapter_out_template: str = "adapters/{model_name}_{timestamp}",
) -> TrainingConfig:
    """Auto-configure training parameters."""
    param_count, base_config = detect_model_params(model_id)

    if train_data:
        dataset_size = _estimate_dataset_size(train_data)
        base_config = _adjust_for_dataset(base_config, dataset_size)

    from datetime import datetime

    model_name = model_id.replace("/", "_").replace("-", "_")
    adapter_out = adapter_out_template.format(
        model_name=model_name,
        timestamp=int(datetime.now().timestamp()),
    )

    return TrainingConfig(
        base_model=model_id,
        train_data=train_data or "data/train.txt",
        adapter_out=adapter_out,
        target_s=base_config["target_s"],
        kp=0.8,
        ki=0.15,
        lora_r=base_config["lora_r"],
        lora_alpha=base_config["lora_alpha"],
        lora_dropout=0.05,
        batch_size=base_config["batch_size"],
        lr=base_config["lr"],
        block_size=base_config["block_size"],
        fp16=True,
        use_unsloth=False,
    )


def _estimate_dataset_size(data_path: str) -> int:
    try:
        if data_path.startswith(("http://", "https://")):
            return 1_000_000
        file_size = Path(data_path).stat().st_size
        return int(file_size * 0.75)
    except Exception:
        return 1_000_000


def _adjust_for_dataset(config: Dict, dataset_size: int) -> Dict:
    adjusted = config.copy()

    if dataset_size < 1_000_000:
        adjusted["target_s"] *= 0.7
        adjusted["batch_size"] = min(adjusted["batch_size"], 4)
        adjusted["lora_r"] = min(adjusted["lora_r"], 16)
    elif dataset_size > 10_000_000:
        adjusted["target_s"] *= 1.2

    return adjusted
