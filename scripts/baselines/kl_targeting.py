#!/usr/bin/env python3
"""
Baseline: Adaptive KL-targeting controller for λ.

Idea: Adjust λ to maintain KL(model || base) ≈ KL* over the batch. This mirrors PPO/SAC-style
temperature controllers and provides a strong adaptive baseline vs SCU's S*-targeted control.

Status: Reference implementation sketch. Not optimized; for small runs.
"""
import math
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def kl_divergence(p_logits, q_logits):
    """KL(p||q) over vocabulary per token, averaged over positions and batch."""
    p_logprobs = torch.log_softmax(p_logits, dim=-1)
    q_logprobs = torch.log_softmax(q_logits, dim=-1)
    p_probs = p_logprobs.exp()
    kl = (p_probs * (p_logprobs - q_logprobs)).sum(dim=-1)  # [batch, seq]
    return kl.mean()  # scalar


@dataclass
class Cfg:
    base_model: str = "meta-llama/Llama-3.2-1B"
    kl_target: float = 0.02
    kp: float = 0.5
    ki: float = 0.1
    deadband: float = 0.002
    lmin: float = 1e-4
    lmax: float = 2.0
    steps: int = 100
    block_size: int = 512
    batch_size: int = 1


def update_lambda_kl(lmbda, kl_meas, kl_target, I, cfg: Cfg):
    e = kl_meas.item() - kl_target
    if abs(e) <= cfg.deadband:
        return lmbda, I
    I = max(-0.2, min(0.2, cfg.ki * e + 0.995 * I))
    lmbda = max(cfg.lmin, min(cfg.lmax, lmbda * math.exp(cfg.kp * e + I)))
    return lmbda, I


def main(cfg: Cfg):
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(cfg.base_model),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
                   target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    )
    model.train()

    # dummy batch (replace with real data loader)
    ids = tok("Adaptive KL baseline test.", return_tensors="pt").input_ids
    attn = torch.ones_like(ids)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    lmbda, I = 1.0, 0.0

    for step in range(cfg.steps):
        with torch.no_grad():
            base_out = base(input_ids=ids, attention_mask=attn)
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        kl = kl_divergence(out.logits[:, :-1], base_out.logits[:, :-1])
        loss = out.loss + lmbda * kl

        opt.zero_grad(); loss.backward(); opt.step()
        lmbda, I = update_lambda_kl(lmbda, kl, cfg.kl_target, I, cfg)
        if step % 10 == 0:
            print(f"step {step} | KL={kl.item():.4f} target {cfg.kl_target:.4f} | λ={lmbda:.4f}")


if __name__ == "__main__":
    main(Cfg())

