"""Capability retention: base vs tuned accuracy on a fixed GSM8K subset +
small MMLU slice (~300 items). Acceptance: within ~1-2 points of base.
See BUILD_PLAN.md section 7d.
"""
from __future__ import annotations

import argparse
from typing import Optional


def run_capability_eval(adapter_path: Optional[str], gsm8k_n: int, mmlu_n: int) -> dict:
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the capability-retention eval.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--adapter", default=None)
    parser.parse_args()
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7d")
