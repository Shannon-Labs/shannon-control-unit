from dataclasses import dataclass
from typing import List
from pathlib import Path
import json
import sqlite3

import psutil
import requests
import torch

from scu_api.config import TrainingConfig


@dataclass
class ValidationIssue:
    message: str
    is_error: bool = False
    suggestion: str = ""


class ValidationReport:
    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues
        self.passed = not any(i.is_error for i in issues)

    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.is_error]

    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if not i.is_error]


class ConfigValidator:
    """Validates training configuration before job submission."""

    def __init__(self):
        self.allowed_models = [
            "gpt2",
            "distilgpt2",
            "microsoft/DialoGPT",
            "facebook/blenderbot",
            "EleutherAI/gpt-neo",
            "EleutherAI/pythia",
            "Llama-3.2",
            "Llama-3.1",
            "google/gemma",
            "microsoft/Phi",
            "tiiuae/falcon",
        ]

    def validate(self, config: TrainingConfig) -> ValidationReport:
        """Run all validation checks."""

        issues: List[ValidationIssue] = []

        # Critical checks (errors)
        issues.extend(self._check_gpu_memory(config))
        issues.extend(self._check_dataset_accessible(config))
        issues.extend(self._check_output_directory(config))

        # Warnings (non-blocking)
        issues.extend(self._check_model_whitelist(config))
        issues.extend(self._check_hyperparameters(config))
        issues.extend(self._check_system_resources(config))

        return ValidationReport(issues)

    def _check_gpu_memory(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Estimate GPU memory requirements."""
        issues: List[ValidationIssue] = []

        if not torch.cuda.is_available():
            return issues

        model_size_gb = self._estimate_model_size(config.base_model)
        dtype_bytes = 2 if config.fp16 else 4
        model_memory_gb = model_size_gb * dtype_bytes

        # LoRA overhead
        lora_overhead_gb = (config.lora_r * model_size_gb * 0.001) + 0.5

        # Batch memory (activations)
        batch_memory_gb = model_memory_gb * (config.batch_size / 4.0) * 0.3

        total_needed = model_memory_gb + lora_overhead_gb + batch_memory_gb

        try:
            props = torch.cuda.get_device_properties(0)
            reserved = torch.cuda.memory_reserved(0)
            available_gb = (props.total_memory - reserved) / (1024**3)
        except Exception as e:
            return [
                ValidationIssue(
                    message=f"Could not query GPU: {e}",
                    is_error=False,
                    suggestion="Check GPU availability",
                )
            ]

        memory_ratio = total_needed / available_gb

        if memory_ratio > 0.95:
            issues.append(
                ValidationIssue(
                    message=f"Requires {total_needed:.1f}GB but only {available_gb:.1f}GB available",
                    is_error=True,
                    suggestion=f"Reduce batch_size to {max(config.batch_size // 2, 1)} or lora_r to {max(config.lora_r // 2, 4)}",
                )
            )
        elif memory_ratio > 0.8:
            issues.append(
                ValidationIssue(
                    message=f"High memory usage: {total_needed:.1f}GB needed",
                    is_error=False,
                    suggestion="Monitor GPU memory",
                )
            )

        return issues

    def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in GB."""
        model_id_lower = model_id.lower()
        size_map = {
            "gpt2": 0.5,
            "distilgpt2": 0.3,
            "microsoft/dialogpt-small": 0.5,
            "microsoft/dialogpt-medium": 1.0,
            "microsoft/dialogpt-large": 2.0,
            "llama-3.2-1": 4.0,
            "llama-3.2-3": 8.0,
            "llama-3.1-8": 16.0,
        }

        for pattern, size in size_map.items():
            if pattern in model_id_lower:
                return size

        import re

        match = re.search(r"(\d+)(?:b|B)", model_id)
        if match:
            return int(match.group(1)) * 2.0

        return 1.0

    def _check_dataset_accessible(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Verify training data is accessible."""
        issues: List[ValidationIssue] = []
        data_path = config.train_data

        if isinstance(data_path, str) and data_path.startswith(("http://", "https://")):
            try:
                response = requests.head(data_path, timeout=5)
                if response.status_code >= 400:
                    issues.append(
                        ValidationIssue(
                            message=f"URL returned {response.status_code}",
                            is_error=True,
                            suggestion="Check URL and access permissions",
                        )
                    )
            except Exception as e:
                issues.append(
                    ValidationIssue(
                        message=f"Cannot access URL: {e}",
                        is_error=True,
                        suggestion="Check network and URL format",
                    )
                )
            return issues

        if not Path(data_path).exists():
            issues.append(
                ValidationIssue(
                    message=f"File not found: {data_path}",
                    is_error=True,
                    suggestion="Provide absolute path or check working directory",
                )
            )
            return issues

        if Path(data_path).stat().st_size == 0:
            issues.append(
                ValidationIssue(
                    message="Dataset file is empty",
                    is_error=True,
                    suggestion="Generate training data first",
                )
            )

        return issues

    def _check_output_directory(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Verify output directory is writable."""
        issues: List[ValidationIssue] = []
        output_dir = Path(config.adapter_out).parent

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".scu_write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(
                ValidationIssue(
                    message=f"Cannot write to {output_dir}: {e}",
                    is_error=True,
                    suggestion="Check directory permissions",
                )
            )

        return issues

    def _check_model_whitelist(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check model compatibility."""
        issues: List[ValidationIssue] = []
        model_id = config.base_model.lower()

        if not any(allowed.lower() in model_id for allowed in self.allowed_models):
            issues.append(
                ValidationIssue(
                    message=f"Model {config.base_model} not in compatibility list",
                    is_error=False,
                    suggestion="May work but not tested with gpt2, llama, etc.",
                )
            )

        return issues

    def _check_hyperparameters(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Validate hyperparameter ranges."""
        issues: List[ValidationIssue] = []

        if config.lr > 1e-3:
            issues.append(
                ValidationIssue(
                    message=f"LR {config.lr} is high for fine-tuning",
                    is_error=False,
                    suggestion="Typical range: 1e-5 to 1e-4",
                )
            )

        if config.target_s > 0.05:
            issues.append(
                ValidationIssue(
                    message=f"Target S {config.target_s} is high (>5%)",
                    is_error=False,
                    suggestion="Typical range: 0.005-0.03",
                )
            )
        elif config.target_s < 0.001:
            issues.append(
                ValidationIssue(
                    message=f"Target S {config.target_s} is low (<0.1%)",
                    is_error=False,
                    suggestion="Typical minimum: 0.005",
                )
            )

        return issues

    def _check_system_resources(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check CPU and RAM resources."""
        issues: List[ValidationIssue] = []

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < 2.0:
            issues.append(
                ValidationIssue(
                    message=f"Low RAM: {available_gb:.1f}GB available",
                    is_error=False,
                    suggestion="Close other applications",
                )
            )

        if (not torch.cuda.is_available()) and config.fp16:
            issues.append(
                ValidationIssue(
                    message="FP16 requested but CUDA not available",
                    is_error=False,
                    suggestion="Set fp16=false for CPU training",
                )
            )

        return issues
