# Resist Sycophancy: SFT + DPO, then a SimPO Comparison

Preference-tune a small open instruct model to hold a correct answer when a
user pushes back with a confident wrong belief, instead of caving. A short
optional SFT stage is followed by a **hand-written DPO objective**, then a
**hand-written SimPO objective** for a controlled head-to-head comparison.
Both losses are implemented by hand (not TRL's trainers) and unit-tested
against TRL's reference implementations on a toy batch.

The full spec this project is built against lives in
[`BUILD_PLAN.md`](BUILD_PLAN.md) — read it for the complete methodology,
scope discipline, and stage-by-stage acceptance checks. Architecture notes
and implementation gotchas for future work sessions live in
[`CLAUDE.md`](CLAUDE.md).

**Status:** scaffold only (Stage 0 of `BUILD_PLAN.md`). Nothing has been
trained or evaluated yet; most modules under `src/` are stubs that raise
`NotImplementedError` pointing at the stage that implements them.

## Repository layout

```
configs/     sft.yaml, dpo.yaml, simpo.yaml, eval.yaml
data/        seeds -> raw pairs -> audited train/test splits (Stage 3)
src/
  data/      synthetic pair generation + splitting
  model/     base + LoRA loading, sequence log-prob primitive
  train/     sft.py, dpo.py (hand-written), simpo.py (hand-written)
  eval/      sycophancy hold-rate + stubbornness guard, LLM judge, capability
  demo/      Gradio side-by-side (base vs tuned, later base/DPO/SimPO)
scripts/     run_generate.sh, run_sft.sh, run_dpo.sh, run_simpo.sh,
             run_eval.sh, run_compare.sh
results/     before_after.md, dpo_vs_simpo.md, figures/
tests/       test_logprobs.py, test_dpo_loss.py, test_simpo_loss.py
             (TRL is used only in the latter two, to verify the hand-written losses)
```

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY and HF_TOKEN
```

## Reproduction (once implemented, in order)

```bash
scripts/run_generate.sh   # Stage 3: synthetic data + audited split
scripts/run_sft.sh        # Stage 4: optional
scripts/run_dpo.sh        # Stage 5
scripts/run_eval.sh outputs/dpo   # Stage 6, fills results/before_after.md
scripts/run_simpo.sh      # Stage 8: only after the DPO path is complete
scripts/run_compare.sh    # Stage 9, fills results/dpo_vs_simpo.md
```

## Method, in one paragraph

DPO scores a response by how much more the trainable policy prefers it than
a frozen reference model does. SimPO drops that reference model entirely and
instead uses the response's length-normalized average log-probability as the
implicit reward, with an explicit target margin the chosen response must
beat the rejected one by. Same paired preference data, two different
objectives — see `BUILD_PLAN.md` sections 5 and 5b for the exact loss forms.

## Results

- [`results/before_after.md`](results/before_after.md) — base vs DPO-tuned
  (hold-rate, flip-rate, stubbornness guard, capability retention).
- [`results/dpo_vs_simpo.md`](results/dpo_vs_simpo.md) — DPO vs SimPO
  head-to-head on identical data and evals.

Both are placeholders until Stages 6 and 9 run.