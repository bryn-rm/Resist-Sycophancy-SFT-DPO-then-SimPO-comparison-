#!/usr/bin/env bash
# Stage 5: train the hand-written DPO objective.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m train.dpo --config configs/dpo.yaml
