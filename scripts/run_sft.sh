#!/usr/bin/env bash
# Stage 4 (optional): light SFT warmup, only if the base model does not
# reliably produce the target two-turn format.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m train.sft --config configs/sft.yaml
