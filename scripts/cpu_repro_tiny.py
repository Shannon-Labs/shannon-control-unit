#!/usr/bin/env python3
"""
CPU demo: PI-controlled MDL regularization on a tiny GPT-2 with LoRA.

Purpose: Show S tracking and lambda updates in minutes on CPU.

Requirements: transformers, peft, torch
"""
import os
import math
from pathlib import Path
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from scu.control import (
    calculate_data_bpt,
    calculate_param_bpt,
    calculate_s_ratio,
    update_lambda,
)


@dataclass
class DemoCfg:
    base_model: str = "sshleifer/tiny-gpt2"
    target_s: float = 0.01
    kp: float = 0.8
    ki: float = 0.15
    deadband: float = 0.002
    lmin: float = 1e-4
    lmax: float = 10.0
    prior_sigma: float = 0.01
    steps: int = 40
    block_size: int = 128
    batch_size: int = 2
    seed: int = 42


def tiny_texts():
    return [
        "Information theory meets control to guide learning.",
        "Adaptive regularization targets a small information ratio.",
        "Closed-loop training reduces both bits-per-token and perplexity.",
        "Simple PI control updates lambda to maintain stability.",
        "No manual sweeps; the loop finds an operating point automatically.",
    ]


def main(cfg: DemoCfg):
    torch.manual_seed(cfg.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model)

    # Add LoRA to GPT-2 tiny. Target common GPT-2 projection names.
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
        inference_mode=False,
    )
    model = get_peft_model(base, peft_cfg)
    model.train()

    # Prepare toy data
    enc = tok(
        tiny_texts(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.block_size,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    tokens_per_epoch = cfg.block_size * cfg.batch_size

    # Simple optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # Control states
    lmbda = 1.0
    I = 0.0
    S_hat = None

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    csv = (out_dir / "scu_cpu_demo.csv").open("w")
    csv.write("step,data_bpt,param_bpt,total_bpt,S,lambda,I\n")

    for step in range(cfg.steps):
        # Simple cyclic batch
        start = (step * cfg.batch_size) % input_ids.size(0)
        end = start + cfg.batch_size
        ids = input_ids[start:end]
        mask = attn[start:end]
        labels = ids.clone()

        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        ce_nats = out.loss
        data_bpt = calculate_data_bpt(ce_nats.item())

        param_bpt = calculate_param_bpt(
            model, sigma=cfg.prior_sigma, tokens_per_epoch=tokens_per_epoch
        )
        S_meas = calculate_s_ratio(data_bpt, param_bpt)

        lmbda, I, S_hat = update_lambda(
            lmbda,
            S_meas,
            cfg.target_s,
            I,
            Kp=cfg.kp,
            Ki=cfg.ki,
            deadband=cfg.deadband,
            lmin=cfg.lmin,
            lmax=cfg.lmax,
            S_hat=S_hat,
        )

        # Convert ParamBPT back to a reg term (nats)
        reg_loss = param_bpt * math.log(2) * tokens_per_epoch
        total = ce_nats + lmbda * reg_loss

        opt.zero_grad()
        total.backward()
        opt.step()

        total_bpt = data_bpt + param_bpt
        csv.write(
            f"{step},{data_bpt:.5f},{param_bpt:.6f},{total_bpt:.6f},{S_meas:.5f},{lmbda:.5f},{I:.5f}\n"
        )
        if step % 5 == 0:
            print(
                f"step {step:03d} | S={S_meas:.4f} (target {cfg.target_s:.4f}) | ",
                f"λ={lmbda:.4f} | DataBPT={data_bpt:.4f} | ParamBPT={param_bpt:.6f}",
            )

    csv.close()
    print("\nCSV written to logs/scu_cpu_demo.csv")


if __name__ == "__main__":
    main(DemoCfg())

