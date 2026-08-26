"""Dedupe, audit-gate, and split raw_pairs.jsonl into train/test (Stage 3).

Split by seed id so no seed leaks across splits. Forces a hand-audit gate:
sample ~50 pairs to a review file and require sign-off before proceeding.
See BUILD_PLAN.md section 6.
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Build audited train/test splits from data/raw_pairs.jsonl.")
    parser.add_argument("--raw", default="data/raw_pairs.jsonl")
    parser.add_argument("--train-out", default="data/train.jsonl")
    parser.add_argument("--test-out", default="data/test.jsonl")
    parser.add_argument("--audit-sample", type=int, default=50)
    parser.parse_args()
    raise NotImplementedError("Stage 3 - see BUILD_PLAN.md section 6")


if __name__ == "__main__":
    main()
