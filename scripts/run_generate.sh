#!/usr/bin/env bash
# Stage 3: generate raw chosen/rejected pairs from data/seeds.jsonl,
# then build the audited train/test split.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m data.generate_pairs --seeds data/seeds.jsonl --out data/raw_pairs.jsonl
python -m data.build_splits --raw data/raw_pairs.jsonl --train-out data/train.jsonl --test-out data/test.jsonl
