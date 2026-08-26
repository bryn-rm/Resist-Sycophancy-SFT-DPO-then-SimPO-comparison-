#!/usr/bin/env bash
# Stage 2 / Stage 6: run the sycophancy + capability eval harness against a
# single adapter (or the base model if --adapter is omitted).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ADAPTER="${1:-}"
ADAPTER_ARGS=()
if [[ -n "$ADAPTER" ]]; then
  ADAPTER_ARGS=(--adapter "$ADAPTER")
fi

python -m eval.sycophancy --config configs/eval.yaml "${ADAPTER_ARGS[@]}"
python -m eval.capability --config configs/eval.yaml "${ADAPTER_ARGS[@]}"
