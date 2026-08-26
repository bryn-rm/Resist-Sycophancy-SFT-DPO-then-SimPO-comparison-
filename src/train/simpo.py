"""Hand-written SimPO loss + training loop (Stage 8) -- added last, after DPO
works end to end. No reference model anywhere in this file. TRL (CPOTrainer,
loss_type="simpo") is only used in tests/test_simpo_loss.py. See BUILD_PLAN.md
section 5b.
"""
from __future__ import annotations

import argparse

import torch


def simpo_loss(
    policy_chosen_mean_logps: torch.Tensor,
    policy_rejected_mean_logps: torch.Tensor,
    beta: float = 2.5,
    gamma: float = 1.0,
) -> dict:
    """logits = beta * (avg_logp_w - avg_logp_l) - gamma; loss = -log_sigmoid(logits).mean()

    avg_logp_* is length-normalized (sequence_logprob(..., mean=True)).
    Returns {loss, reward_w, reward_l, reward_margin, pref_accuracy}.
    """
    raise NotImplementedError("Stage 8 - see BUILD_PLAN.md section 5b")


def train(config_path: str):
    raise NotImplementedError("Stage 8 - see BUILD_PLAN.md section 5b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SimPO LoRA adapter.")
    parser.add_argument("--config", default="configs/simpo.yaml")
    args = parser.parse_args()
    train(args.config)
