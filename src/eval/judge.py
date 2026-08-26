"""LLM-as-judge (Claude) classifying second-turn replies as holds / caves /
updates-appropriately, plus validation against hand labels. Report
judge-vs-human agreement (target > 90%) in the README. See BUILD_PLAN.md
section 7c.
"""
from __future__ import annotations

import argparse


def judge_response(question: str, gold_answer: str, wrong_pushback: str, reply: str) -> str:
    """Returns one of: 'holds', 'caves', 'updates_appropriately'."""
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7c")


def validate_judge(hand_labels_path: str, n: int = 60) -> float:
    """Hand-label ~60 responses, return judge-vs-human agreement."""
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7c")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the LLM judge against hand labels.")
    parser.add_argument("--n", type=int, default=60)
    parser.parse_args()
    raise NotImplementedError("Stage 2 - see BUILD_PLAN.md section 7c")
