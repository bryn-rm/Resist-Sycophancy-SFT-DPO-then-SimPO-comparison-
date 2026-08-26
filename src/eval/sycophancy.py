"""Two-turn pushback eval (Stage 2, build BEFORE training). Two modes:

- hold-rate / flip-rate (7a): held-out correct-answer questions, does the
  second-turn reply hold under wrong pushback.
- stubbornness guard (7b): turn-1 answer is WRONG and pushback is CORRECT,
  does the model appropriately update. This is the honesty check.

Classification is done by src/eval/judge.py, not string matching.
See BUILD_PLAN.md section 7a/7b.
"""
from __future__ import annotations

import argparse
from typing import Optional


def run_hold_rate_eval(test_path: str, adapter_path: Optional[str]) -> dict:
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7a")


def run_stubbornness_guard(test_path: str, adapter_path: Optional[str]) -> dict:
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the sycophancy hold-rate + stubbornness-guard evals.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--adapter", default=None, help="Path to a LoRA adapter, or omit for the base model.")
    parser.parse_args()
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7a/7b")
