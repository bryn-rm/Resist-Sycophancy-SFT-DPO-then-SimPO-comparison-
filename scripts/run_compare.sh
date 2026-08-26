#!/usr/bin/env bash
# Stage 9: run the identical eval harness against base, DPO, and SimPO
# adapters and emit results/dpo_vs_simpo.md. Keep everything else (data,
# eval, seeds) fixed so the objective is the only variable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Evaluating base model..."
./scripts/run_eval.sh

echo "Evaluating DPO adapter..."
./scripts/run_eval.sh outputs/dpo

echo "Evaluating SimPO adapter..."
./scripts/run_eval.sh outputs/simpo

echo "TODO (Stage 9): assemble results/dpo_vs_simpo.md from the three runs above."
