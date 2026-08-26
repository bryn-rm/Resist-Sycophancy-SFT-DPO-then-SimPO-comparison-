"""Synthetic chosen/rejected pair generation via the Anthropic API (Stage 3).

For each seed in data/seeds.jsonl: draft a correct first answer, a realistic
wrong pushback, and both a holding (chosen) and caving (rejected) second
response. Keep the provider behind an interface so it can be swapped.
See BUILD_PLAN.md section 6.
"""
from __future__ import annotations

import argparse
from typing import Protocol


class PairGenerator(Protocol):
    """Thin interface so the LLM provider can be swapped without touching callers."""

    def generate(self, question: str, gold_answer: str) -> dict:
        ...


class AnthropicPairGenerator:
    def __init__(self, model: str = "claude-sonnet-5", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def generate(self, question: str, gold_answer: str) -> dict:
        raise NotImplementedError("Stage 3 - see BUILD_PLAN.md section 6")


def main():
    parser = argparse.ArgumentParser(description="Generate raw chosen/rejected pairs from data/seeds.jsonl.")
    parser.add_argument("--seeds", default="data/seeds.jsonl")
    parser.add_argument("--out", default="data/raw_pairs.jsonl")
    parser.parse_args()
    raise NotImplementedError("Stage 3 - see BUILD_PLAN.md section 6")


if __name__ == "__main__":
    main()
