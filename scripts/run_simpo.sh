#!/usr/bin/env bash
# Stage 8: train the hand-written SimPO objective. Only run after the DPO
# path (through Stage 7) is complete and working.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m train.simpo --config configs/simpo.yaml
