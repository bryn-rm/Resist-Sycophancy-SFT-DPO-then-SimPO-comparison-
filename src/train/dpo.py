"""Hand-written DPO loss + training loop (Stage 5). Do NOT use TRL's
DPOTrainer here -- TRL is only allowed in tests/test_dpo_loss.py, to verify
this implementation. See BUILD_PLAN.md section 5.

Reference model: the SAME PeftModel with the LoRA adapter disabled
(`model.disable_adapter()`), not a second model in memory.
"""
from __future__ import annotations

import argparse

import torch


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> dict:
    """Sigmoid-form DPO loss (+ optional cDPO label smoothing).

    logratio_w = policy_chosen_logps - ref_chosen_logps
    logratio_l = policy_rejected_logps - ref_rejected_logps
    logits = beta * (logratio_w - logratio_l)
    loss = -log_sigmoid(logits).mean()  [label_smoothing == 0]

    Returns {loss, reward_w, reward_l, reward_margin, pref_accuracy}.
    """
    raise NotImplementedError("Stage 5 - see BUILD_PLAN.md section 5")


def train(config_path: str):
    raise NotImplementedError("Stage 5 - see BUILD_PLAN.md section 5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DPO LoRA adapter.")
    parser.add_argument("--config", default="configs/dpo.yaml")
    args = parser.parse_args()
    train(args.config)
