from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from shannon_control import control, data
from scu_api.config import TrainingConfig


@dataclass
class TrainingMetrics:
    step: int
    data_bpt: float
    param_bpt: float
    total_bpt: float
    s_ratio: float
    lambda_value: float
    loss: float
    tokens_per_second: float
    eta_minutes: float


def _setup_device_and_dtype(fp16: bool) -> Tuple[str, torch.dtype, bool]:
    """Detect device and dtype; prefer CUDA with 4-bit when available."""

    if torch.cuda.is_available():
        return "cuda", torch.float16 if fp16 else torch.float32, True
    if torch.backends.mps.is_available():
        return "mps", torch.float32, False
    return "cpu", torch.float32, False


class TrainingEngine:
    """Minimal, callable training engine backed by the reference SCU loop."""

    def __init__(self, config: TrainingConfig, job_id: str = "job"):
        self.config = config
        self.job_id = job_id
        self.logger = logging.getLogger(f"scu_api.training.{job_id}")
        self.accelerator: Optional[Accelerator] = None
        self.device = None
        self.dtype = None
        self.use_4bit = False

    def run(self, progress_callback: Optional[Callable[[TrainingMetrics], None]] = None) -> Path:
        """Execute training synchronously; returns adapter directory."""

        self._set_seed(self.config.seed)
        self.device, self.dtype, self.use_4bit = _setup_device_and_dtype(self.config.fp16)

        self.accelerator = Accelerator(
            mixed_precision="fp16" if (self.config.fp16 and torch.cuda.is_available()) else None,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        model, tokenizer = self._load_model_and_tokenizer()
        train_chunks, actual_block_size = self._prepare_data(tokenizer)
        tokens_per_epoch = len(train_chunks) * actual_block_size

        optimizer = AdamW(model.parameters(), lr=self.config.lr, weight_decay=0.0)
        num_training_steps = self._compute_training_steps(len(train_chunks))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * num_training_steps)),
            num_training_steps=num_training_steps,
        )

        model, optimizer, scheduler = self.accelerator.prepare(model, optimizer, scheduler)

        lmbda = self.config.lambda_init
        I = 0.0
        S_hat = None
        global_step = 0
        data_iter = data.create_data_iterator(train_chunks, self.config.batch_size)
        start_time = time.time()
        last_log_ts = start_time

        csv_file = None
        csv_writer = None
        if self.config.log_csv:
            csv_path = Path(self.config.log_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(csv_path, "w", newline="")
            import csv

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "step",
                "data_bpt",
                "param_bpt",
                "total_bpt",
                "S",
                "lambda",
                "I",
                "wall_time_s",
            ])

        self.logger.info(
            "Starting training | steps=%s | target_s=%.4f | base_model=%s",
            num_training_steps,
            self.config.target_s,
            self.config.base_model,
        )

        for epoch in range(self.config.epochs if not self.config.steps else 1):
            if self.config.steps and global_step >= self.config.steps:
                break

            for batch_chunks in data_iter:
                if self.config.steps and global_step >= self.config.steps:
                    break

                batch_ids = torch.tensor([c["input_ids"] for c in batch_chunks])
                batch_mask = torch.tensor([c["attention_mask"] for c in batch_chunks])

                batch_ids = batch_ids.to(self.accelerator.device)
                batch_mask = batch_mask.to(self.accelerator.device)
                labels = batch_ids.clone()

                outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=labels)
                ce_loss_nats = outputs.loss
                data_bpt = control.calculate_data_bpt(ce_loss_nats.item())
                param_bpt = control.calculate_param_bpt(
                    model,
                    sigma=self.config.prior_sigma,
                    tokens_per_epoch=tokens_per_epoch,
                )
                s_ratio = control.calculate_s_ratio(data_bpt, param_bpt)

                lmbda, I, S_hat = control.update_lambda(
                    lmbda,
                    s_ratio,
                    self.config.target_s,
                    I,
                    Kp=self.config.kp,
                    Ki=self.config.ki,
                    deadband=self.config.deadband,
                    lmin=self.config.lambda_min,
                    lmax=self.config.lambda_max,
                    S_hat=S_hat,
                )

                reg_loss = param_bpt * math.log(2) * tokens_per_epoch
                total_loss = ce_loss_nats + lmbda * reg_loss

                self.accelerator.backward(total_loss)

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if csv_writer:
                    total_bpt = data_bpt + param_bpt
                    csv_writer.writerow(
                        [
                            global_step,
                            f"{data_bpt:.6f}",
                            f"{param_bpt:.6f}",
                            f"{total_bpt:.6f}",
                            f"{s_ratio:.6f}",
                            f"{lmbda:.6f}",
                            f"{I:.6f}",
                            f"{time.time() - start_time:.2f}",
                        ]
                    )
                    csv_file.flush()

                elapsed = time.time() - start_time
                tokens_seen = (global_step + 1) * self.config.batch_size * actual_block_size
                tps = tokens_seen / max(elapsed, 1e-6)
                steps_left = max(num_training_steps - (global_step + 1), 0)
                eta_minutes = (steps_left * (elapsed / max(global_step + 1, 1))) / 60

                metrics = TrainingMetrics(
                    step=global_step,
                    data_bpt=data_bpt,
                    param_bpt=param_bpt,
                    total_bpt=data_bpt + param_bpt,
                    s_ratio=s_ratio,
                    lambda_value=lmbda,
                    loss=total_loss.item(),
                    tokens_per_second=tps,
                    eta_minutes=eta_minutes,
                )

                if progress_callback:
                    progress_callback(metrics)

                if time.time() - last_log_ts > 10:
                    self.logger.info(
                        "step=%d data_bpt=%.4f param_bpt=%.6f S=%.3f lambda=%.4f",  # noqa: E501
                        global_step,
                        data_bpt,
                        param_bpt,
                        s_ratio,
                        lmbda,
                    )
                    last_log_ts = time.time()

                global_step += 1

        if csv_file:
            csv_file.close()

        adapter_dir = self._save_adapter(model, tokenizer)
        self.logger.info("Finished training -> %s", adapter_dir)
        return adapter_dir

    def _load_model_and_tokenizer(self):
        # Optional Unsloth fast loader path
        if self.config.use_unsloth:
            try:
                from unsloth import FastLanguageModel  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "use_unsloth=True but unsloth is not installed; run `pip install unsloth`"
                ) from exc

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                load_in_4bit=True,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            FastLanguageModel.for_training(
                model,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
        else:
            quantization_config = None
            if self.use_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )

            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=quantization_config,
                torch_dtype=self.dtype,
                device_map="auto" if self.device != "cpu" else None,
                trust_remote_code=True,
            )
            try:
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
            except Exception:
                pass

            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                inference_mode=False,
            )

            model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
        return model, tokenizer

    def _prepare_data(self, tokenizer) -> Tuple[list, int]:
        train_texts = data.load_texts_from_file(
            self.config.train_data,
            max_texts=self.config.max_texts,
        )

        if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length > 0:
            block_size = min(self.config.block_size, tokenizer.model_max_length)
        else:
            block_size = self.config.block_size

        train_chunks = data.tokenize_and_chunk(
            train_texts,
            tokenizer,
            block_size=block_size,
            shuffle=True,
            seed=self.config.seed,
        )

        return train_chunks, block_size

    def _compute_training_steps(self, num_chunks: int) -> int:
        if self.config.steps:
            return self.config.steps
        steps_per_epoch = max(num_chunks // self.config.batch_size, 1)
        return max(1, self.config.epochs * steps_per_epoch)

    def _save_adapter(self, model, tokenizer) -> Path:
        output_dir = Path(self.config.adapter_out)
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        metadata = {
            "job_id": self.job_id,
            "base_model": self.config.base_model,
            "target_s": self.config.target_s,
            "kp": self.config.kp,
            "ki": self.config.ki,
            "deadband": self.config.deadband,
            "lambda_init": self.config.lambda_init,
            "lambda_min": self.config.lambda_min,
            "lambda_max": self.config.lambda_max,
            "epochs": self.config.epochs,
            "steps": self.config.steps,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "block_size": self.config.block_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return output_dir

    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
