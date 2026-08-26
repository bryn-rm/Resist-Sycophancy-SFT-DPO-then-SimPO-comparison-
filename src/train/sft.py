"""Optional light SFT stage (Stage 4) -- LoRA, short, only if the base model
does not reliably produce the target two-turn format. See configs/sft.yaml
and BUILD_PLAN.md section 8 (Stage 4).
"""
from __future__ import annotations

import argparse


def train(config_path: str):
    raise NotImplementedError("Stage 4 - see BUILD_PLAN.md section 8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optional SFT warmup.")
    parser.add_argument("--config", default="configs/sft.yaml")
    args = parser.parse_args()
    train(args.config)
