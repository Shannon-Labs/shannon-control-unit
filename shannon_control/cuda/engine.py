"""
CUDA Training Engine for Shannon Control Unit

PyTorch + CUDA training pipeline with SCU PI control, optimized for
NVIDIA DGX Spark (Grace Blackwell, 128GB unified memory).
"""

import json
import math
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import CUDATrainingConfig
from .callback import SCUCUDACallback


class JSONLDataset(Dataset):
    """Simple JSONL dataset for causal language modeling."""

    def __init__(self, path: str, tokenizer, block_size: int, text_key: str = "text"):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.chunks = []

        # Load and tokenize
        texts = []
        with open(path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get(text_key, "") or data.get("content", "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue

        # Concatenate all text and chunk into block_size sequences
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        # Create fixed-length chunks
        for i in range(0, len(all_tokens) - block_size, block_size):
            chunk = all_tokens[i : i + block_size]
            self.chunks.append(chunk)

        self.total_tokens = len(all_tokens)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class CUDATrainingEngine:
    """
    CUDA training engine with SCU PI control.

    Provides a complete training pipeline for NVIDIA GPUs:
    - Model loading with optional quantization (4-bit, 8-bit, or full precision)
    - LoRA application via PEFT
    - SCU PI controller for adaptive regularization
    - MDL saturation detection for self-terminating training
    - GPU memory and utilization monitoring

    Example:
        from shannon_control.cuda import CUDATrainingEngine, CUDATrainingConfig

        config = CUDATrainingConfig(
            base_model="Qwen/Qwen3.5-27B",
            train_data="data/fineweb_edu_1gb.jsonl",
            target_s=0.04,
            steps=2000,
        )

        engine = CUDATrainingEngine(config)
        adapter_path = engine.run()
    """

    def __init__(self, config: CUDATrainingConfig, job_id: str = "cuda_job"):
        self.config = config
        self.job_id = job_id
        self.model = None
        self.tokenizer = None
        self.scu_callback = None

    def run(
        self,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> Path:
        """
        Execute training and return adapter path.

        Args:
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to saved adapter directory.
        """
        self._check_prerequisites()

        print(f"[CUDA-SCU] Starting training job: {self.job_id}")
        print(f"[CUDA-SCU] Model: {self.config.base_model}")
        print(f"[CUDA-SCU] Target S: {self.config.target_s}")
        print(f"[CUDA-SCU] Device: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA-SCU] GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        # Set seeds
        self._set_seed(self.config.seed)

        # Enable TF32 for Blackwell/Ampere+
        if self.config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load model
        print("[CUDA-SCU] Loading model...")
        self.model, self.tokenizer = self._load_model()

        # Apply LoRA
        print("[CUDA-SCU] Applying LoRA...")
        self.model = self._apply_lora()

        # Load data
        print(f"[CUDA-SCU] Loading training data from {self.config.train_data}...")
        train_dataset = JSONLDataset(
            self.config.train_data,
            self.tokenizer,
            self.config.block_size,
        )
        tokens_per_epoch = train_dataset.total_tokens
        print(f"[CUDA-SCU] Total tokens: {tokens_per_epoch:,}")
        print(f"[CUDA-SCU] Training chunks: {len(train_dataset):,}")

        val_dataset = None
        if self.config.val_data:
            val_dataset = JSONLDataset(
                self.config.val_data,
                self.tokenizer,
                self.config.block_size,
            )

        # Create SCU callback
        self.scu_callback = SCUCUDACallback(
            model=self.model,
            tokens_per_epoch=tokens_per_epoch,
            target_s=self.config.target_s,
            control_frequency=self.config.control_frequency,
            Kp=self.config.kp,
            Ki=self.config.ki,
            deadband=self.config.deadband,
            lambda_init=self.config.lambda_init,
            lambda_min=self.config.lambda_min,
            lambda_max=self.config.lambda_max,
            prior_sigma=self.config.prior_sigma,
            log_dir=self.config.log_dir,
            log_csv=self.config.log_csv,
        )

        # Run training
        print("[CUDA-SCU] Starting training loop...")
        start_time = time.time()

        try:
            self._train(train_dataset, val_dataset, progress_callback)
        finally:
            self.scu_callback.shutdown()

        elapsed = time.time() - start_time
        print(f"[CUDA-SCU] Training completed in {elapsed / 60:.1f} minutes")

        # Save adapter
        adapter_path = self._save_adapter()
        print(f"[CUDA-SCU] Adapter saved to: {adapter_path}")

        # Print summary
        summary = self.scu_callback.get_metrics_summary()
        print("\n[CUDA-SCU] Training Summary:")
        print(f"  Final S-ratio: {summary.get('final_s_ratio', 0):.4f}")
        print(f"  Final lambda:  {summary.get('final_lambda', 0):.4f}")
        print(f"  Final loss:    {summary.get('final_loss', 0):.4f}")
        print(f"  Saturated:     {summary.get('saturated', False)}")

        return adapter_path

    def _check_prerequisites(self):
        """Check that CUDA and required packages are available."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This engine requires an NVIDIA GPU.\n"
                "DGX Spark should have CUDA available via the Blackwell GPU."
            )

        try:
            import transformers
            import peft
        except ImportError as e:
            raise RuntimeError(
                f"Missing required package: {e}\n"
                "Install with: pip install -r requirements-cuda.txt"
            )

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_model(self):
        """Load model and tokenizer with optional quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Quantization config
        quantization_config = None
        if self.config.use_4bit:
            torch_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Determine dtype
        if self.config.dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif self.config.dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": True,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, **model_kwargs
        )

        # Disable cache for training
        model.config.use_cache = False

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Memory report
        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[CUDA-SCU] Model loaded. GPU memory used: {mem_gb:.1f} GB")

        return model, tokenizer

    def _apply_lora(self):
        """Apply LoRA layers via PEFT."""
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        # Prepare for quantized training if needed
        if self.config.use_4bit or self.config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            inference_mode=False,
        )

        model = get_peft_model(self.model, peft_config)
        model.print_trainable_parameters()

        return model

    def _train(
        self,
        train_dataset: JSONLDataset,
        val_dataset: Optional[JSONLDataset],
        progress_callback: Optional[Callable] = None,
    ):
        """Run the training loop with SCU control."""
        from transformers import get_cosine_schedule_with_warmup

        # Dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            drop_last=True,
        )

        # Optimizer — weight_decay MUST be 0 (SCU provides regularization)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=0.0,
        )

        # Scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.steps,
        )

        # Optional torch.compile
        if self.config.compile_model:
            print("[CUDA-SCU] Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

        # Mixed precision
        use_amp = self.config.dtype in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if self.config.dtype == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(self.config.dtype == "fp16"))

        # Training
        self.model.train()
        global_step = 0
        accum_loss = 0.0
        data_iter = iter(train_loader)

        print(f"[CUDA-SCU] Training for {self.config.steps} steps")
        print(f"[CUDA-SCU] Effective batch size: {self.config.effective_batch_size}")

        while global_step < self.config.steps:
            optimizer.zero_grad()

            # Gradient accumulation
            for accum_step in range(self.config.gradient_accumulation_steps):
                # Get batch (cycle through data)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()

                # Forward pass with AMP
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    ce_loss = outputs.loss

                    # SCU regularization (differentiable)
                    reg_loss = self.scu_callback.compute_regularization_loss()
                    total_loss = ce_loss + reg_loss

                    # Scale for accumulation
                    total_loss = total_loss / self.config.gradient_accumulation_steps

                scaler.scale(total_loss).backward()
                accum_loss += ce_loss.item() / self.config.gradient_accumulation_steps

            # Optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # SCU callback
            self.scu_callback.on_step(global_step, accum_loss)

            # Periodic logging
            if global_step % self.config.report_steps == 0:
                lam = self.scu_callback.get_current_lambda()
                s = self.scu_callback.metrics_history[-1].s_ratio if self.scu_callback.metrics_history else 0
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[SCU] Step {global_step}: "
                    f"loss={accum_loss:.4f}, S={s:.4f} (target={self.config.target_s}), "
                    f"lambda={lam:.4f}, lr={lr:.2e}, mem={mem_gb:.1f}GB"
                )

            # MDL saturation check
            if self.scu_callback.is_saturated() and global_step > self.config.steps // 4:
                print(f"\n[SCU] MDL SATURATION DETECTED at step {global_step}!")
                print(f"[SCU] Lambda has stabilized — model has learned all meaningful patterns.")
                print(f"[SCU] Stopping training early (would have run to step {self.config.steps}).")
                break

            # Progress callback
            if progress_callback:
                progress_callback({
                    "step": global_step,
                    "total_steps": self.config.steps,
                    "loss": accum_loss,
                    "lambda": self.scu_callback.get_current_lambda(),
                })

            accum_loss = 0.0
            global_step += 1

        # Validation pass
        if val_dataset:
            self._validate(val_dataset, amp_dtype, use_amp)

    def _validate(self, val_dataset, amp_dtype, use_amp):
        """Run validation pass."""
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].cuda()
                labels = batch["labels"].cuda()

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    n_batches += 1

                if n_batches >= 50:
                    break

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[CUDA-SCU] Validation loss: {avg_loss:.4f}")
        self.model.train()

    def _save_adapter(self) -> Path:
        """Save trained adapter and metadata."""
        output_dir = Path(self.config.adapter_out)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PEFT adapter
        unwrapped = self.model
        if hasattr(self.model, "module"):
            unwrapped = self.model.module
        unwrapped.save_pretrained(str(output_dir))

        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(output_dir))

        # Save SCU metrics
        if self.scu_callback:
            self.scu_callback.save_metrics(str(output_dir / "scu_metrics.json"))

        # Save metadata
        summary = self.scu_callback.get_metrics_summary() if self.scu_callback else {}
        metadata = {
            "job_id": self.job_id,
            "base_model": self.config.base_model,
            "framework": "pytorch-cuda",
            "hardware": "dgx-spark",
            "architecture": "SCU-PI-Control",
            "scu_config": {
                "target_s": self.config.target_s,
                "kp": self.config.kp,
                "ki": self.config.ki,
                "deadband": self.config.deadband,
                "lambda_init": self.config.lambda_init,
                "lambda_min": self.config.lambda_min,
                "lambda_max": self.config.lambda_max,
            },
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "target_modules": self.config.lora_target_modules,
            },
            "training_config": {
                "steps": self.config.steps,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "effective_batch_size": self.config.effective_batch_size,
                "lr": self.config.lr,
                "block_size": self.config.block_size,
                "dtype": self.config.dtype,
                "seed": self.config.seed,
            },
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return output_dir
