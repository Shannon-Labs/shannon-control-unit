"""
MLX Training Engine for Shannon Control Unit

Provides a complete native MLX training pipeline with SCU PI control integration.
This engine uses mlx-lm's training infrastructure while adding SCU metrics
and adaptive regularization.
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any

from .config import MLXTrainingConfig
from .callback import SCUTrainingCallback


class MLXTrainingEngine:
    """
    Native MLX training engine with SCU control.

    This engine provides:
    - Native MLX model loading and LoRA application
    - Integration with mlx-lm's training loop
    - SCU PI control via TrainingCallback
    - Apple Silicon power monitoring
    - Comprehensive metrics logging

    Example:
        from shannon_control.mlx import MLXTrainingEngine, MLXTrainingConfig

        config = MLXTrainingConfig(
            base_model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            train_data="data/train.jsonl",
            steps=1000
        )

        engine = MLXTrainingEngine(config)
        adapter_path = engine.run()
    """

    def __init__(self, config: MLXTrainingConfig, job_id: str = "mlx_job"):
        """
        Initialize MLX Training Engine.

        Args:
            config: MLX training configuration
            job_id: Unique identifier for this training job
        """
        self.config = config
        self.job_id = job_id
        self.model = None
        self.tokenizer = None
        self.scu_callback = None

    def run(
        self,
        progress_callback: Optional[Callable[[dict], None]] = None
    ) -> Path:
        """
        Execute training and return adapter path.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Path to saved adapter directory
        """
        # Check MLX availability
        if not self._check_mlx_available():
            raise RuntimeError(
                "MLX is not available. This engine requires Apple Silicon. "
                "Install with: pip install mlx mlx-lm"
            )

        print(f"[MLX-SCU] Starting training job: {self.job_id}")
        print(f"[MLX-SCU] Model: {self.config.base_model}")
        print(f"[MLX-SCU] Target S: {self.config.target_s}")

        # Set random seed
        self._set_seed(self.config.seed)

        # Load model and tokenizer
        print("[MLX-SCU] Loading model...")
        self.model, self.tokenizer = self._load_model()

        # Apply LoRA (or train full parameters)
        if self.config.train_full_params:
            print("[MLX-SCU] Full-parameter mode enabled (no LoRA)")
            self._ensure_full_trainable_params()
        else:
            print("[MLX-SCU] Applying LoRA...")
            self._apply_lora()

        # Estimate tokens per epoch
        tokens_per_epoch = self._estimate_tokens_per_epoch()
        print(f"[MLX-SCU] Estimated tokens per epoch: {tokens_per_epoch:,}")

        # Create SCU callback
        log_csv = self.config.log_csv or (
            str(Path(self.config.log_dir) / "metrics.csv")
            if self.config.log_dir else None
        )

        self.scu_callback = SCUTrainingCallback(
            model=self.model,
            tokens_per_epoch=tokens_per_epoch,
            target_s=self.config.target_s,
            control_frequency=self.config.control_frequency,
            param_scope=self.config.param_scope,
            param_bpt_frequency=self.config.param_bpt_frequency,
            Kp=self.config.kp,
            Ki=self.config.ki,
            deadband=self.config.deadband,
            lambda_init=self.config.lambda_init,
            lambda_min=self.config.lambda_min,
            lambda_max=self.config.lambda_max,
            prior_sigma=self.config.prior_sigma,
            enable_power_monitoring=self.config.enable_power_monitoring,
            log_dir=self.config.log_dir,
            log_csv=log_csv,
        )

        # Run training
        print("[MLX-SCU] Starting training loop...")
        start_time = time.time()

        try:
            self._train(progress_callback)
        finally:
            # Ensure clean shutdown
            self.scu_callback.shutdown()

        elapsed = time.time() - start_time
        print(f"[MLX-SCU] Training completed in {elapsed/60:.1f} minutes")

        # Save adapter
        adapter_path = self._save_adapter()
        print(f"[MLX-SCU] Adapter saved to: {adapter_path}")

        # Print summary
        summary = self.scu_callback.get_metrics_summary()
        print("\n[MLX-SCU] Training Summary:")
        print(f"  Final S-ratio: {summary.get('final_s_ratio', 0):.4f}")
        print(f"  Final lambda: {summary.get('final_lambda', 0):.4f}")
        print(f"  Final loss: {summary.get('final_loss', 0):.4f}")

        return adapter_path

    def _check_mlx_available(self) -> bool:
        """Check if MLX is available."""
        try:
            import mlx.core
            import mlx_lm
            return True
        except ImportError:
            return False

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

        try:
            import mlx.core as mx
            mx.random.seed(seed)
        except Exception:
            pass

    def _load_model(self):
        """Load model and tokenizer using mlx-lm."""
        from mlx_lm import load

        model, tokenizer = load(self.config.base_model)
        return model, tokenizer

    def _apply_lora(self):
        """Apply LoRA layers to the model."""
        from mlx_lm.tuner.utils import linear_to_lora_layers

        # mlx-lm API: linear_to_lora_layers(model, num_layers, config, use_dora=False)
        lora_config = {
            "rank": self.config.lora_r,
            "scale": self.config.lora_alpha / self.config.lora_r,
            "dropout": self.config.lora_dropout,
        }

        linear_to_lora_layers(
            self.model,
            num_layers=self.config.lora_layers,
            config=lora_config,
        )

        # Count trainable parameters
        trainable = 0
        if hasattr(self.model, 'trainable_parameters'):
            params = self.model.trainable_parameters()
            from mlx.utils import tree_flatten
            flat = tree_flatten(params)
            trainable = sum(p.size for _, p in flat)
        print(f"[MLX-SCU] Trainable parameters: {trainable:,}")

    def _ensure_full_trainable_params(self):
        """Ensure training uses full parameter set when LoRA is disabled."""
        if hasattr(self.model, "parameters"):
            if hasattr(self.model, "trainable_parameters"):
                self.model.trainable_parameters = self.model.parameters
        try:
            from mlx.utils import tree_flatten
            params = self.model.parameters() if hasattr(self.model, "parameters") else self.model.trainable_parameters()
            flat = tree_flatten(params)
            trainable = sum(p.size for _, p in flat)
            print(f"[MLX-SCU] Trainable parameters (full): {trainable:,}")
        except Exception:
            pass

    def _estimate_tokens_per_epoch(self) -> int:
        """Estimate total tokens per epoch from training data."""
        train_path = Path(self.config.train_data)

        if not train_path.exists():
            print(f"[MLX-SCU] Warning: Training data not found at {train_path}")
            return 100000  # Default estimate

        # Count approximate tokens
        total_chars = 0
        try:
            with open(train_path, 'r') as f:
                if train_path.suffix == '.jsonl':
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '') or data.get('content', '')
                            total_chars += len(text)
                        except json.JSONDecodeError:
                            continue
                else:
                    total_chars = len(f.read())
        except Exception as e:
            print(f"[MLX-SCU] Warning: Could not read training data: {e}")
            return 100000

        # Rough estimate: ~4 chars per token
        estimated_tokens = total_chars // 4

        # Adjust based on actual training (steps * batch * block)
        max_tokens = (
            self.config.steps *
            self.config.effective_batch_size *
            self.config.block_size
        )

        return min(estimated_tokens, max_tokens) or 100000

    def _load_jsonl_data(self, path: str):
        """Load JSONL data from file."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return data

    def _train(self, progress_callback: Optional[Callable] = None):
        """
        Run the training loop using mlx-lm's train function.
        """
        import mlx.optimizers as optim
        from mlx_lm.tuner.trainer import train, TrainingArgs
        from mlx_lm.tuner.datasets import TextDataset, CacheDataset

        # Load training data
        print(f"[MLX-SCU] Loading training data from {self.config.train_data}...")
        train_data = self._load_jsonl_data(self.config.train_data)
        print(f"[MLX-SCU] Loaded {len(train_data)} training samples")

        # Create TextDataset and wrap with CacheDataset for preprocessing
        text_dataset = TextDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            text_key="text",
        )
        train_set = CacheDataset(text_dataset)

        # Load validation data if provided
        val_set = []
        if self.config.val_data:
            print(f"[MLX-SCU] Loading validation data from {self.config.val_data}...")
            val_data = self._load_jsonl_data(self.config.val_data)
            print(f"[MLX-SCU] Loaded {len(val_data)} validation samples")
            val_text_dataset = TextDataset(
                data=val_data,
                tokenizer=self.tokenizer,
                text_key="text",
            )
            val_set = CacheDataset(val_text_dataset)

        # Create optimizer
        optimizer = optim.AdamW(learning_rate=self.config.lr)

        # Create output directory
        output_dir = Path(self.config.adapter_out)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        # Enable validation if val_set is provided, use 25 batches for quick eval
        val_batches = 25 if val_set else 0
        eval_steps = self.config.eval_steps if self.config.eval_steps > 0 and val_set else self.config.steps

        adapter_file = output_dir / ("adapters.safetensors" if not self.config.train_full_params else "model.safetensors")
        train_args = TrainingArgs(
            batch_size=self.config.batch_size,
            iters=self.config.steps,
            val_batches=val_batches,
            steps_per_report=self.config.report_steps,
            steps_per_eval=eval_steps,
            steps_per_save=self.config.save_steps,
            max_seq_length=self.config.block_size,
            adapter_file=str(adapter_file),
            grad_checkpoint=self.config.grad_checkpoint,
        )

        # Run training
        train(
            model=self.model,
            optimizer=optimizer,
            train_dataset=train_set,
            val_dataset=val_set if val_set else [],
            args=train_args,
            training_callback=self.scu_callback,
        )

    def _save_adapter(self) -> Path:
        """Save the trained adapter or full model and metadata."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        output_dir = Path(self.config.adapter_out)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.train_full_params:
            model_file = output_dir / "model.safetensors"
            if not model_file.exists():
                weights = {}
                if hasattr(self.model, "parameters"):
                    flat_params = dict(tree_flatten(self.model.parameters()))
                    for name, param in flat_params.items():
                        mx.eval(param)
                        weights[name] = param
                if weights:
                    mx.save_safetensors(str(model_file), weights)
        else:
            # Save adapter weights (mlx-lm should have done this, but ensure it's there)
            adapter_file = output_dir / "adapters.safetensors"
            if not adapter_file.exists():
                # Manual save if needed
                weights = {}
                if hasattr(self.model, 'trainable_parameters'):
                    flat_params = dict(tree_flatten(self.model.trainable_parameters()))
                    for name, param in flat_params.items():
                        if "lora" in name.lower():
                            mx.eval(param)
                            weights[name] = param
                    if weights:
                        mx.save_safetensors(str(adapter_file), weights)

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(output_dir))

        # Save SCU metrics
        if self.scu_callback:
            self.scu_callback.save_metrics(str(output_dir / "scu_metrics.json"))

        # Save metadata
        metadata = {
            "job_id": self.job_id,
            "base_model": self.config.base_model,
            "framework": "mlx",
            "training_mode": "full" if self.config.train_full_params else "lora",
            "scu_config": {
                "target_s": self.config.target_s,
                "kp": self.config.kp,
                "ki": self.config.ki,
                "deadband": self.config.deadband,
                "lambda_init": self.config.lambda_init,
                "param_scope": self.config.param_scope,
                "param_bpt_frequency": self.config.param_bpt_frequency,
            },
            "training_config": {
                "steps": self.config.steps,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "lora_r": self.config.lora_r if not self.config.train_full_params else None,
                "lora_alpha": self.config.lora_alpha if not self.config.train_full_params else None,
            },
            "summary": self.scu_callback.get_metrics_summary() if self.scu_callback else {},
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return output_dir


def quick_train(
    model_id: str = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
    train_data: str = "data/train.jsonl",
    steps: int = 100,
    **kwargs
) -> Path:
    """
    Quick training helper for interactive use.

    Args:
        model_id: MLX model to fine-tune
        train_data: Path to training data
        steps: Number of training steps
        **kwargs: Additional config overrides

    Returns:
        Path to saved adapter

    Example:
        >>> from shannon_control.mlx import quick_train
        >>> adapter = quick_train(steps=50)
    """
    config = MLXTrainingConfig(
        base_model=model_id,
        train_data=train_data,
        steps=steps,
        **kwargs
    )
    engine = MLXTrainingEngine(config)
    return engine.run()
