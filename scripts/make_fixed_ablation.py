#!/usr/bin/env python3
"""
Make a short fixed-λ ablation CSV on CPU (Tiny GPT-2 + LoRA).

Outputs: ablations/fixed_5.0.csv by default with header:
  step,data_bpt,param_bpt,S,lambda,I

Usage:
  python scripts/make_fixed_ablation.py --lambda_val 5.0 --steps 60 --out ablations/fixed_5.0.csv

Notes:
  - Keep runs tiny to finish quickly in CI or CPU-only environments.
  - Requires: transformers, peft, torch. The CI workflow pins versions to avoid TF imports.
"""
import argparse
import math
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from scu.control import calculate_data_bpt, calculate_param_bpt, calculate_s_ratio


def run(lambda_val: float, steps: int, out_path: str, block_size: int = 128, batch_size: int = 2, sigma: float = 0.01):
    # Avoid TensorFlow import paths in some transformers versions
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    base_id = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_id)
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

    texts = [
        "Information theory meets control to guide learning.",
        "Adaptive regularization targets a small information ratio.",
        "Closed-loop training reduces both bits-per-token and perplexity.",
        "Simple PI control updates lambda to maintain stability.",
        "No manual sweeps; the loop finds an operating point automatically.",
    ]
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=block_size)
    ids, attn = enc["input_ids"], enc["attention_mask"]

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    tpe = block_size * batch_size

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        f.write("step,data_bpt,param_bpt,S,lambda,I\n")
        for step in range(steps):
            s = (step * batch_size) % ids.size(0)
            e = s + batch_size
            x, m = ids[s:e], attn[s:e]
            labels = x.clone()

            out = model(input_ids=x, attention_mask=m, labels=labels)
            ce_nats = out.loss
            data_bpt = calculate_data_bpt(ce_nats.item())
            param_bpt = calculate_param_bpt(model, sigma=sigma, tokens_per_epoch=tpe)
            S = calculate_s_ratio(data_bpt, param_bpt)

            reg_loss = param_bpt * math.log(2) * tpe
            loss = ce_nats + lambda_val * reg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            f.write(f"{step},{data_bpt:.6f},{param_bpt:.6f},{S:.6f},{lambda_val:.4f},0.0000\n")

    print(f"Wrote {outp} ({outp.stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_val", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", type=str, default="ablations/fixed_5.0.csv")
    args = ap.parse_args()
    run(args.lambda_val, args.steps, args.out)


if __name__ == "__main__":
    main()

