# 🎯 ACTIONABLE ENGINEERING TASKS - SCU Training API

**This is a tactical implementation guide with specific, copy-pasteable code.**

---

## TASK 1: Backend Engineer — Configuration Validation System

**Objective:** Build pre-flight validation to catch 80%+ of failures before training starts.

**File to Create:** `scu_api/service/validator.py`

### Step 1.1: Create Validation Classes

```python
# scu_api/service/validator.py
from dataclasses import dataclass
from typing import List
from pathlib import Path
import torch
import psutil
import requests

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
            "gpt2", "microsoft/DialoGPT", "facebook/blenderbot",
            "EleutherAI/gpt-neo", "EleutherAI/pythia",
            "Llama-3.2", "Llama-3.1", "gemma",
            "stanford-crfm/BioMedLM", "microsoft/Phi"
        ]
    
    def validate(self, config: TrainingConfig) -> ValidationReport:
        """Run all validation checks."""
        issues = []
        
        # Critical checks (errors)
        issues.extend(self._check_gpu_memory(config))
        issues.extend(self._check_dataset_accessible(config))
        issues.extend(self._check_output_directory(config))
        
        # Warning checks (non-blocking)
        issues.extend(self._check_model_whitelist(config))
        issues.extend(self._check_hyperparameters(config))
        issues.extend(self._check_lora_configuration(config))
        issues.extend(self._check_system_resources(config))
        
        return ValidationReport(issues)
    
    def _check_gpu_memory(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Estimate GPU memory requirements."""
        issues = []
        
        if not torch.cuda.is_available():
            return issues
        
        # Estimate model size (rough approximation)
        model_size_gb = self._estimate_model_size(config.base_model)
        
        # Memory components
        # 1. Model weights
        dtype_bytes = 2 if config.fp16 else 4
        model_memory_gb = model_size_gb * dtype_bytes
        
        # 2. LoRA overhead
        lora_overhead_gb = (config.lora_r * model_size_gb * 0.001) + 0.5
        
        # 3. Activations (rough estimate)
        batch_memory_gb = model_memory_gb * (config.batch_size / 4.0) * 0.3
        
        total_needed = model_memory_gb + lora_overhead_gb + batch_memory_gb
        
        # Get available GPU memory
        gpu_id = 0  # Primary GPU
        try:
            props = torch.cuda.get_device_properties(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id)
            reserved = torch.cuda.memory_reserved(gpu_id)
            available_gb = (props.total_memory - reserved) / (1024**3)
        except Exception as e:
            issues.append(ValidationIssue(
                message=f"Could not query GPU memory: {e}",
                is_error=False,
                suggestion="Ensure GPU is accessible and nvidia-smi works"
            ))
            return issues
        
        # Check if we have enough memory
        memory_ratio = total_needed / available_gb
        
        if memory_ratio > 0.95:
            issues.append(ValidationIssue(
                message=f"High memory usage: {total_needed:.1f}GB needed, {available_gb:.1f}GB available",
                is_error=True,
                suggestion=f"Reduce batch_size to {max(config.batch_size//2, 1)} or lora_r to {max(config.lora_r//2, 4)}"
            ))
        elif memory_ratio > 0.8:
            issues.append(ValidationIssue(
                message=f"Memory usage may be high: {total_needed:.1f}GB needed",
                is_error=False,
                suggestion="Monitor GPU memory during training"
            ))
        
        return issues
    
    def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in GB."""
        model_id_lower = model_id.lower()
        
        # Common model sizes
        size_map = {
            "gpt2": 0.5,
            "distilgpt2": 0.3,
            "microsoft/DialoGPT-small": 0.5,
            "microsoft/DialoGPT-medium": 1.0,
            "microsoft/DialoGPT-large": 2.0,
            "llama-3.2-1": 4.0,
            "llama-3.2-3": 8.0,
            "llama-3.1-8": 16.0,
        }
        
        for pattern, size in size_map.items():
            if pattern.lower() in model_id_lower:
                return size
        
        # Try to extract from model name (e.g., "7b", "13b")
        import re
        match = re.search(r'(\d+)(?:b|B)', model_id)
        if match:
            param_billion = int(match.group(1))
            # Rough estimate: 2GB per billion parameters (fp16)
            return param_billion * 2.0
        
        # Default fallback
        return 4.0  # Assume 1-2B model if unknown
    
    def _check_dataset_accessible(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Verify training data is accessible."""
        issues = []
        
        data_path = config.train_data
        
        # Check if it's a URL
        if data_path.startswith(("http://", "https://")):
            try:
                response = requests.head(data_path, timeout=5)
                if response.status_code >= 400:
                    issues.append(ValidationIssue(
                        message=f"Dataset URL returned status {response.status_code}",
                        is_error=True,
                        suggestion="Check URL and verify it's publicly accessible"
                    ))
            except Exception as e:
                issues.append(ValidationIssue(
                    message=f"Cannot access dataset URL: {e}",
                    is_error=True,
                    suggestion="Check network connectivity and URL format"
                ))
            return issues
        
        # Check local file
        if not Path(data_path).exists():
            issues.append(ValidationIssue(
                message=f"Dataset file not found: {data_path}",
                is_error=True,
                suggestion="Provide absolute path or check working directory"
            ))
            return issues
        
        # Check file is not empty
        if Path(data_path).stat().st_size == 0:
            issues.append(ValidationIssue(
                message="Dataset file is empty",
                is_error=True,
                suggestion="Generate or download training data first"
            ))
        
        return issues
    
    def _check_output_directory(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Verify output directory is writable."""
        issues = []
        
        output_dir = Path(config.adapter_out).parent
        
        try:
            # Try to create directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permission
            test_file = output_dir / ".scu_write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(ValidationIssue(
                message=f"Cannot write to output directory {output_dir}: {e}",
                is_error=True,
                suggestion="Check directory permissions or choose a different location"
            ))
        
        return issues
    
    def _check_model_whitelist(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check model compatibility."""
        issues = []
        
        model_id = config.base_model.lower()
        is_allowed = any(allowed.lower() in model_id for allowed in self.allowed_models)
        
        if not is_allowed:
            issues.append(ValidationIssue(
                message=f"Model {config.base_model} not in compatibility list",
                is_error=False,
                suggestion="May work but not tested. Consider using: gpt2, llama-3.2, etc."
            ))
        
        return issues
    
    def _check_hyperparameters(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Validate hyperparameter ranges."""
        issues = []
        
        # Learning rate check
        if config.lr > 1e-3:
            issues.append(ValidationIssue(
                message=f"Learning rate {config.lr} is high for fine-tuning",
                is_error=False,
                suggestion="Typical range: 1e-5 to 1e-4. Consider 5e-5"
            ))
        
        # Target S check
        if config.target_s > 0.05:
            issues.append(ValidationIssue(
                message=f"Target S {config.target_s} is high (>5%)",
                is_error=False,
                suggestion="Typical range: 0.005 to 0.03. May cause underfitting"
            ))
        elif config.target_s < 0.001:
            issues.append(ValidationIssue(
                message=f"Target S {config.target_s} is low (<0.1%)",
                is_error=False,
                suggestion="Typical minimum: 0.005. May cause overfitting"
            ))
        
        return issues
    
    def _check_lora_configuration(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Validate LoRA settings."""
        issues = []
        
        # LoRA rank too high for small models
        if config.lora_r > 32 and self._estimate_model_size(config.base_model) < 2.0:
            issues.append(ValidationIssue(
                message=f"LoRA rank {config.lora_r} may be high for this model",
                is_error=False,
                suggestion="Reduce to 8-16 for models < 2B parameters"
            ))
        
        return issues
    
    def _check_system_resources(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check CPU and RAM resources."""
        issues = []
        
        # Check available RAM
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 2.0:
            issues.append(ValidationIssue(
                message=f"Low RAM: {available_gb:.1f}GB available",
                is_error=False,
                suggestion="Close other applications or use machine with more RAM"
            ))
        
        # Check if using CPU (warning)
        if not torch.cuda.is_available() and config.fp16:
            issues.append(ValidationIssue(
                message="FP16 requested but CUDA not available",
                is_error=False,
                suggestion="Set fp16=false for CPU training"
            ))
        
        return issues
```

### Step 1.2: Add Validation Endpoint to Server

```python
# Add to scu_api/server.py
from scu_api.service.validator import ConfigValidator, ValidationReport

@app.post("/validate")
async def validate_config(payload: SubmitRequest):
    """Validate training configuration before submission."""
    config = TrainingConfig(
        base_model=payload.base_model,
        train_data=payload.train_data,
        steps=payload.steps,
        target_s=payload.target_s,
        kp=payload.kp,
        ki=payload.ki,
        adapter_out=payload.adapter_out,
        batch_size=payload.batch_size,
        lr=payload.lr,
        fp16=payload.fp16,
        max_texts=payload.max_texts,
        use_unsloth=payload.use_unsloth,
    )
    
    validator = ConfigValidator()
    report = validator.validate(config)
    
    return {
        "valid": report.passed,
        "errors": [
            {"message": i.message, "suggestion": i.suggestion}
            for i in report.errors()
        ],
        "warnings": [
            {"message": i.message, "suggestion": i.suggestion}
            for i in report.warnings()
        ]
    }
```

### Step 1.3: Integrate Validation into Job Submission

```python
# Update in scu_api/server.py
@app.post("/jobs", response_model=SubmitResponse)
async def submit_job(payload: SubmitRequest):
    # Create config
    config = TrainingConfig(...)  # as before
    
    # Validate before submission
    validator = ConfigValidator()
    report = validator.validate(config)
    
    if not report.passed:
        # Return errors instead of submitting
        error_details = "\n".join([
            f"ERROR: {i.message}" for i in report.errors()
        ])
        raise HTTPException(
            status_code=400,
            detail=f"Configuration validation failed:\n{error_details}"
        )
    
    # If validation passes, submit job
    job_id = jobs.submit(config)
    return SubmitResponse(job_id=job_id, status="queued")
```

### Step 1.4: Test Validation

```python
# tests/test_validator.py
def test_gpu_memory_check():
    """Test GPU memory validation."""
    config = TrainingConfig(
        base_model="gpt2",
        train_data="test.txt",
        adapter_out="adapters/test",
        batch_size=100,  # Ridiculously large
        fp16=True
    )
    validator = ConfigValidator()
    report = validator._check_gpu_memory(config)
    
    # Should return error about high memory usage
    assert any("memory" in i.message.lower() for i in report)
```

Deploy and test:
```bash
# Start server
python -m scu_api.server

# Test with invalid config
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "gpt2",
    "train_data": "/nonexistent/file.txt",
    "steps": 10
  }'

# Should return: 400 error about inaccessible dataset
```

**Time Estimate:** 1 day (6-8 hours)  
**Deliverable:** Working validation system that prevents 60%+ of failures  

---

## TASK 2: ML Engineer — Smart Configuration System

**Objective:** Make the API "just work" without manual tuning.

**File to Create:** `scu_api/service/smart_config.py`

### Step 2.1: Create Model Database

```python
# scu_api/service/smart_config.py
MODEL_PARAMETERS = {
    # GPT variants
    "gpt2": 0.124,
    "distilgpt2": 0.082,
    "microsoft/DialoGPT-small": 0.124,
    "microsoft/DialoGPT-medium": 0.355,
    "microsoft/DialoGPT-large": 0.774,
    
    # Llama 3.2 family
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
    "microsoft/Phi-3-mini": 3.8,
    "microsoft/Phi-3-small": 7.0,
    
    # Gemma
    "google/gemma-2b": 2.0,
    "google/gemma-7b": 7.0,
    
    # Other common models
    "facebook/blenderbot-400m-distill": 0.4,
    "facebook/blenderbot-1B-distill": 1.0,
    "facebook/blenderbot-3B": 3.0,
    "tiiuae/falcon-7b": 7.0,
    "tiiuae/falcon-40b": 40.0,
}

CONFIG_PRESETS = {
    # Format: (min_params_GB, max_params_GB): {config}
    (0.0, 0.2): {
        "target_s": 0.005,
        "lora_r": 8,
        "lora_alpha": 16,
        "batch_size": 8,
        "lr": 1e-4,
        "block_size": 512,
    },
    (0.2, 1.0): {
        "target_s": 0.008,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 4,
        "lr": 8e-5,
        "block_size": 1024,
    },
    (1.0, 4.0): {
        "target_s": 0.01,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,
        "lr": 5e-5,
        "block_size": 2048,
    },
    (4.0, 10.0): {
        "target_s": 0.02,
        "lora_r": 32,
        "lora_alpha": 64,
        "batch_size": 1,
        "lr": 3e-5,
        "block_size": 2048,
    },
    (10.0, 100.0): {
        "target_s": 0.03,
        "lora_r": 64,
        "lora_alpha": 128,
        "batch_size": 1,
        "lr": 2e-5,
        "block_size": 4096,
    },
}
```

### Step 2.2: Implement Model Detection

```python
import re
from typing import Tuple

def detect_model_size(model_id: str) -> Tuple[float, Dict[str, any]]:
    """
    Detect model size in billions of parameters and return optimal config.
    
    Returns: (param_count_GB, config_dict)
    """
    model_id_lower = model_id.lower()
    
    # Direct match
    for known_id, params in MODEL_PARAMETERS.items():
        if known_id.lower() in model_id_lower:
            return params, get_config_for_params(params)
    
    # Pattern match for common formats
    patterns = [
        # Format: ...-7b-...
        (r'[-_](\d+)(?:b|B)(?:[-_]|$)', 1.0),
        # Format: ...-70b-...
        (r'[-_](\d{2,3})(?:b|B)(?:[-_]|$)', 1.0),
        # Format: ...70b...
        (r'(\d{2,3})b', 1.0),
    ]
    
    for pattern, multiplier in patterns:
        match = re.search(pattern, model_id)
        if match:
            param_count = int(match.group(1)) * multiplier
            return param_count, get_config_for_params(param_count)
    
    # Default fallback (assume small model for safety)
    return 1.0, CONFIG_PRESETS[(0.2, 1.0)]


def get_config_for_params(param_count: float) -> Dict[str, any]:
    """Get config preset for parameter count."""
    for (min_params, max_params), config in CONFIG_PRESETS.items():
        if min_params <= param_count < max_params:
            return config.copy()
    
    # Default to largest category
    return CONFIG_PRESETS[(10.0, 100.0)].copy()
```

### Step 2.3: Build Auto-Configurator

```python
from pathlib import Path

def auto_configure(
    model_id: str,
    train_data_path: Optional[str] = None,
    adapter_out_template: str = "adapters/{model_name}_{timestamp}"
) -> TrainingConfig:
    """
    Automatically configure training parameters for a model.
    
    Args:
        model_id: HuggingFace model ID
        train_data_path: Path to training data (optional, for dataset-aware tuning)
        adapter_out_template: Template for adapter output directory
    
    Returns:
        Complete TrainingConfig object
    """
    # Get base config from model size
    param_count, base_config = detect_model_size(model_id)
    
    # Adjust for dataset size if provided
    if train_data_path:
        dataset_size = _estimate_dataset_size(train_data_path)
        base_config = _adjust_for_dataset(base_config, dataset_size)
    
    # Generate output directory name
    from datetime import datetime
    model_name = model_id.replace("/", "_").replace("-", "_")
    adapter_out = adapter_out_template.format(
        model_name=model_name,
        timestamp=int(datetime.now().timestamp())
    )
    
    # Build complete config
    return TrainingConfig(
        base_model=model_id,
        train_data=train_data_path or "data/train.txt",
        adapter_out=adapter_out,
        # Control parameters
        target_s=base_config["target_s"],
        kp=0.8,
        ki=0.15,
        # LoRA
        lora_r=base_config["lora_r"],
        lora_alpha=base_config["lora_alpha"],
        lora_dropout=0.05,
        # Training
        batch_size=base_config["batch_size"],
        lr=base_config["lr"],
        block_size=base_config["block_size"],
        # Hardware
        fp16=True,
        use_unsloth=False,  # Let user enable if they want
    )


def _estimate_dataset_size(data_path: str) -> int:
    """Estimate dataset size in tokens (rough)."""
    try:
        if data_path.startswith(("http://", "https://")):
            # For URLs, assume medium size
            return 1_000_000
        
        file_size = Path(data_path).stat().st_size
        # Rough: 1 token ≈ 0.75 bytes
        return int(file_size * 0.75)
    except:
        return 1_000_000  # Default assumption


def _adjust_for_dataset(config: Dict, dataset_size: int) -> Dict:
    """Adjust config based on dataset size."""
    adjusted = config.copy()
    
    # Small dataset (< 1M tokens) - more conservative
    if dataset_size < 1_000_000:
        adjusted["target_s"] *= 0.7  # More regularization
        adjusted["batch_size"] = min(adjusted["batch_size"], 4)
        adjusted["lora_r"] = min(adjusted["lora_r"], 16)
    
    # Large dataset (> 10M tokens) - can be more aggressive
    elif dataset_size > 10_000_000:
        adjusted["target_s"] *= 1.2  # Less regularization
        adjusted["batch_size"] = min(adjusted["batch_size"], 8)
    
    return adjusted
```

### Step 2.4: Add Intelligence to Server

```python
# Add to scu_api/server.py
from scu_api.service.smart_config import auto_configure

class AutoConfigRequest(BaseModel):
    model_id: str
    train_data_path: Optional[str] = None


@app.post("/auto-config")
async def get_auto_configuration(request: AutoConfigRequest):
    """Get automatically configured parameters for a model."""
    try:
        config = auto_configure(
            model_id=request.model_id,
            train_data_path=request.train_data_path
        )
        return {
            "model_id": request.model_id,
            "suggested_config": config.dict(),
            "estimated_params_GB": detect_model_size(request.model_id)[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Step 2.5: Test Smart Configuration

```python
# tests/test_smart_config.py

def test_model_size_detection():
    """Test model size detection works for common models."""
    assert detect_model_size("gpt2")[0] == 0.124
    assert detect_model_size("meta-llama/Llama-3.2-3B")[0] == 3.0
    assert detect_model_size("EleutherAI/pythia-2.8b")[0] == 2.8

def test_auto_configure():
    """Test auto-configuration produces valid configs."""
    config = auto_configure("gpt2")
    assert config.lora_r == 8
    assert config.target_s == 0.005
    assert config.batch_size == 8
    
    config = auto_configure("meta-llama/Llama-3.2-3B")
    assert config.lora_r == 32
    assert config.target_s == 0.02

def test_dataset_size_adjustment():
    """Test small datasets get conservative configs."""
    config = auto_configure("gpt2", train_data_path="small_dataset.txt")
    # Small dataset should reduce target_s
    assert config.target_s < 0.005
```

**Time Estimate:** 1.5 days (12 hours)  
**Deliverable:** Working auto-configuration that makes API "just work"  

---

## TASK 3: Full-Stack Engineer — Python SDK & CLI

**Objective:** Make API accessible to Python users and terminal users.

**Files to Create:**
- `scu_api/client/sync_client.py` (Python SDK)
- `scu_api/cli/main.py` (CLI tool)
- `setup.py` (package installer)

### Step 3.1: Create Python SDK

```python
# scu_api/client/sync_client.py
import requests
import time
from typing import Dict, Optional, Iterator
from pathlib import Path
from tqdm import tqdm

class SCUClient:
    """Synchronous client for SCU Training API."""
    
    def __init__(self, api_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def submit_job(
        self,
        base_model: str,
        train_data: str,
        wait: bool = False,
        show_progress: bool = True,
        **config_overrides
    ) -> Dict:
        """
        Submit a training job.
        
        Args:
            base_model: HuggingFace model ID
            train_data: Path or URL to training data
            wait: If True, block until job completes
            show_progress: Show progress bar if waiting
            **config_overrides: Additional configuration parameters
        
        Returns:
            Job information dictionary
        """
        payload = {
            "base_model": base_model,
            "train_data": str(train_data),
            **config_overrides
        }
        
        response = self.session.post(f"{self.api_url}/jobs", json=payload)
        response.raise_for_status()
        
        job_info = response.json()
        
        if wait:
            return self.wait_for_completion(job_info["job_id"], show_progress=show_progress)
        
        return job_info
    
    def wait_for_completion(self, job_id: str, show_progress: bool = True, poll_interval: int = 5) -> Dict:
        """Wait for job to complete and return final status."""
        
        with tqdm(total=100, desc=f"Job {job_id}", disable=not show_progress) as pbar:
            while True:
                status = self.get_job_status(job_id)
                
                if status["status"] == "succeeded":
                    pbar.update(100 - pbar.n)
                    return status
                
                if status["status"] == "failed":
                    raise RuntimeError(f"Job failed: {status.get('error')}")
                
                # Update progress bar
                progress = status.get("progress", 0)
                pbar.update(progress - pbar.n)
                
                time.sleep(poll_interval)
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get current status of a job."""
        response = self.session.get(f"{self.api_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self, status_filter: Optional[str] = None, limit: int = 50) -> list:
        """List jobs with optional filtering."""
        params = {"limit": limit}
        if status_filter:
            params["status"] = status_filter
        
        response = self.session.get(f"{self.api_url}/jobs", params=params)
        response.raise_for_status()
        return response.json()["jobs"]
    
    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a running job."""
        response = self.session.post(f"{self.api_url}/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()
    
    def validate_config(self, config: Dict) -> Dict:
        """Validate configuration before submission."""
        response = self.session.post(f"{self.api_url}/validate", json=config)
        response.raise_for_status()
        return response.json()
    
    def get_auto_config(self, model_id: str, train_data: Optional[str] = None) -> Dict:
        """Get auto-configured parameters for a model."""
        payload = {"model_id": model_id}
        if train_data:
            payload["train_data_path"] = train_data
        
        response = self.session.post(f"{self.api_url}/auto-config", json=payload)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if API server is healthy."""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
```

### Step 3.2: Create CLI Tool

```python
# scu_api/cli/main.py
import click
import json
from pathlib import Path
from scu_api.client.sync_client import SCUClient


@click.group()
@click.option("--api-url", default="http://localhost:8000", help="SCU API URL")
@click.pass_context
def cli(ctx, api_url):
    """SCU Training CLI."""
    ctx.ensure_object(dict)
    ctx.obj["client"] = SCUClient(api_url)


@cli.command()
@click.option("--base-model", required=True, help="HuggingFace model ID")
@click.option("--train-data", required=True, type=click.Path(exists=True), 
              help="Path to training data")
@click.option("--auto-config", is_flag=True, help="Use auto-configuration")
@click.option("--wait", is_flag=True, help="Wait for completion")
@click.option("--val-data", type=click.Path(exists=True), help="Validation data")
@click.argument("config_args", nargs=-1)
@click.pass_context
def train(ctx, base_model, train_data, auto_config, wait, val_data, config_args):
    """Submit a training job."""
    
    client = ctx.obj["client"]
    
    # Check API health
    if not client.health_check():
        click.echo("❌ API server not available", err=True)
        return
    
    click.echo(f"🚀 Training {base_model} on {train_data}")
    
    # Parse additional config from args
    # Format: --lr=5e-5 --batch_size=2
    config_overrides = {}
    for arg in config_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to convert to int/float
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                pass  # Keep as string
            config_overrides[key] = value
    
    # Get auto config if requested
    if auto_config:
        click.echo("🔮 Getting auto-configuration...")
        auto_result = client.get_auto_config(base_model, str(train_data))
        click.echo(f"   Suggested target_s: {auto_result['suggested_config']['target_s']}")
        click.echo(f"   Suggested lora_r: {auto_result['suggested_config']['lora_r']}")
        click.echo(f"   Suggested batch_size: {auto_result['suggested_config']['batch_size']}")
        
        if click.confirm("Use these settings?"):
            # Merge auto-config with user overrides
            config_overrides = {**auto_result['suggested_config'], **config_overrides}
    
    # Validate first
    config = {
        "base_model": base_model,
        "train_data": str(train_data),
        **config_overrides
    }
    
    click.echo("✅ Validating configuration...")
    validation = client.validate_config(config)
    
    if not validation["valid"]:
        click.echo("❌ Validation failed:", err=True)
        for error in validation["errors"]:
            click.echo(f"   ERROR: {error['message']}", err=True)
            if error['suggestion']:
                click.echo(f"   💡 {error['suggestion']}", err=True)
        return
    
    if validation["warnings"]:
        click.echo("⚠️  Warnings:")
        for warning in validation["warnings"]:
            click.echo(f"   {warning['message']}")
            if warning['suggestion']:
                click.echo(f"   💡 {warning['suggestion']}")
        
        if not click.confirm("Continue anyway?"):
            return
    
    # Submit job
    try:
        job = client.submit_job(
            base_model=base_model,
            train_data=str(train_data),
            wait=wait,
            **config_overrides
        )
        
        click.echo(f"✅ Job submitted: {job['job_id']}")
        
        if job["status"] == "succeeded":
            click.echo(f"🎉 Training completed!")
            click.echo(f"   Adapter: {job.get('adapter_path', 'N/A')}")
            if "last_metrics" in job:
                metrics = job["last_metrics"]
                click.echo(f"   Final BPT: {metrics.get('total_bpt', 'N/A'):.3f}")
                click.echo(f"   Final S: {metrics.get('s_ratio', 'N/A'):.4f}")
        
    except Exception as e:
        click.echo(f"❌ Failed to submit job: {e}", err=True)


@cli.command()
@click.argument("job-id")
@click.pass_context
def status(ctx, job_id):
    """Get job status."""
    client = ctx.obj["client"]
    
    try:
        job = client.get_job_status(job_id)
        
        click.echo(f"Job: {job['job_id']}")
        click.echo(f"Status: {job['status']}")
        click.echo(f"Progress: {job.get('progress', 0):.1f}%")
        
        if job.get('adapter_path'):
            click.echo(f"Adapter: {job['adapter_path']}")
        
        if job.get('error'):
            click.echo(f"Error: {job['error']}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.option("--status-filter", type=click.Choice(['queued', 'running', 'succeeded', 'failed']))
@click.option("--limit", default=20, help="Maximum jobs to show")
@click.pass_context
def jobs(ctx, status_filter, limit):
    """List jobs."""
    client = ctx.obj["client"]
    
    try:
        jobs_list = client.list_jobs(status_filter, limit=limit)
        
        if not jobs_list:
            click.echo("No jobs found")
            return
        
        # Print table
        headers = f"{'ID':<10} {'Model':<25} {'Status':<12} {'Progress':<10}"
        click.echo(headers)
        click.echo("-" * len(headers))
        
        for job in jobs_list:
            model_name = job['base_model'].split('/')[-1][:23]
            progress = f"{job.get('progress', 0):.1f}%"
            click.echo(f"{job['job_id']:<10} {model_name:<25} {job['status']:<12} {progress:<10}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.pass_context
def health(ctx):
    """Check API health."""
    client = ctx.obj["client"]
    
    if client.health_check():
        click.echo("✅ API is healthy")
    else:
        click.echo("❌ API is not responding")


if __name__ == "__main__":
    cli()
```

### Step 3.3: Create setup.py for Installation

```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scu-api",
    version="0.1.0",
    author="Hunter Bown",
    author_email="hunter@shannonlabs.dev",
    description="Shannon Control Unit Training API - Adaptive regularization for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shannon-Labs/shannon-control-unit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=2.0.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "server": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.18.0",
            "accelerate>=0.20.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "peft>=0.4.0",
            "bitsandbytes>=0.39.0",
            "unsloth; extra == 'unsloth'",
        ],
    },
    entry_points={
        "console_scripts": [
            "scu=scu_api.cli.main:cli",
        ],
    },
)
```

### Step 3.4: Install and Test

```bash
# Install in development mode
pip install -e .[dev,server]

# Test CLI
scu health
# Should show: ✅ API is healthy

# Test SDK
python3 -c "
from scu_api import SCUClient
client = SCUClient()
print('✅ Client initialized')
print(client.health_check())
"

# Full end-to-end test
scu train \
  --base-model sshleifer/tiny-gpt2 \
  --train-data data/train.txt \
  --steps 5 \
  --lora-r 2 \
  --wait

# Should show progress bar and complete successfully
```

**Time Estimate:** 1 day (6-8 hours)  
**Deliverable:** Installable package with SDK and CLI  

---

## WEEK 1 COMPLETION CHECKLIST

### Day 1-2 (Backend Engineer):
- [ ] `scu_api/service/validator.py` created and working
- [ ] `/validate` endpoint working  
- [ ] Validation integrated into job submission
- [ ] GPU memory check catches OOM scenarios
- [ ] Tests for validation system pass

### Day 3-4 (ML Engineer):
- [ ] `scu_api/service/smart_config.py` created and working
- [ ] `/auto-config` endpoint working
- [ ] Model size detection working for 10+ architectures
- [ ] Dataset-aware adjustments implemented
- [ ] Tests for auto-config pass

### Day 5-6 (Full-Stack Engineer):
- [ ] `scu_api/client/sync_client.py` created with all methods
- [ ] `scu_api/cli/main.py` created with all commands
- [ ] `setup.py` created and tested
- [ ] Package installs with `pip install -e .`
- [ ] End-to-end test passes (submit → train → get adapter)

### Day 7 (Integration):
- [ ] All three components working together
- [ ] Update API_STATUS.md with completion
- [ ] Write integration tests
- [ ] Prepare Week 2 plan (persistence, cancellation, etc.)

---

## 🚀 IMMEDIATE NEXT STEPS (Start Monday)

### 1. Backend Engineer: Configuration Validation
```bash
cd /Volumes/VIXinSSD/shannon-control-unit
mkdir -p scu_api/service
# Create validator.py (paste code from TASK 1 above)
# Add validation endpoint to server.py
# Test with: curl -X POST http://localhost:8000/validate -d '{...}'
```

### 2. ML Engineer: Smart Configuration  
```bash
mkdir -p scu_api/service
# Create smart_config.py (paste code from TASK 2 above)
# Add auto-config endpoint to server.py
# Test model detection: curl -X POST http://localhost:8000/auto-config \
#   -d '{"model_id": "gpt2"}'
```

### 3. Full-Stack Engineer: Python SDK & CLI
```bash
mkdir -p scu_api/client scu_api/cli
# Create sync_client.py and main.py (paste code from TASK 3 above)
# Create setup.py
# Install: pip install -e .[dev,server]
# Test: scu health
```

### 4. Team Lead: Coordination & Testing
- [ ] Review each component
- [ ] Ensure they work together
- [ ] Run end-to-end test
- [ ] Update documentation

**Expected Completion:** 5-6 days  
**Deliverable:** Full API with validation, smart config, Python SDK, and CLI  
**Impact:** Users can train models with zero config expertise
