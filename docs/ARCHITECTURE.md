# Architecture & Design Notes — Sycophancy-DPO

This is the "how it actually works" reference: per-module implementation notes, the rationale behind the non-obvious decisions, and the data flow end to end. `BUILD_PLAN.md` is the spec (what to build and why, stage by stage) — read that first. `CLAUDE.md` is the short, session-critical file (status, gotchas, testing rules); this doc is for when you need the fuller picture behind a specific module or decision. Update it as each stage lands.

## Module notes

### `src/model/load.py`
- `load_tokenizer` / `load_base_model` / `attach_lora` / `load_policy_and_tokenizer`.
- The DPO reference model is **not** a second copy in memory: reference behaviour comes from the same `PeftModel` with the adapter disabled (`peft`'s `disable_adapter()` context), not a second `from_pretrained` call. Pure VRAM decision — see `BUILD_PLAN.md` section 5.
- `load_in_4bit` toggles a `BitsAndBytesConfig` (nf4, double quant, bf16 compute dtype). The LoRA adapter itself is always full precision on top of the quantized base.
- `python -m model.load --base-model Qwen/Qwen2.5-3B-Instruct` is the Stage 0 smoke test (base + fresh LoRA adapter loads, 4-bit optional).

### `src/model/logprobs.py` (Stage 1)
- `sequence_logprob(model, tokenizer, prompt, response, mean=False)` is the one primitive everything downstream is built on. DPO uses `mean=False` (summed log-prob); SimPO uses `mean=True` (length-normalized). Both call the *same* function — do not fork this into two implementations.
- Prompt-region tokens must be masked to `-100` before the loss is computed, or the log-prob silently includes tokens the model didn't generate. Test this explicitly; it's the easiest place to introduce a bug that only surfaces as nonsense reward margins three stages later.
- Logits/labels are offset by one position (next-token prediction) — shift before gathering.

### `src/train/dpo.py` (Stage 5 — not yet implemented)
- Hand-written. Do not import `trl.DPOTrainer` here — TRL is a dev-only dependency, used exclusively in `tests/test_dpo_loss.py` to check this implementation against a reference on a toy batch.
- Reference log-probs are precomputed once under `torch.no_grad()` and cached, since the reference model never changes across training — this roughly halves compute per step. Don't recompute them every batch.
- `beta` default `0.1`. `label_smoothing > 0` switches to the cDPO variant (exact loss form in `BUILD_PLAN.md` section 5).
- Log `reward_margin` and `pref_accuracy` every step, and decode a few samples periodically. Reward margin climbing while sample quality degrades is the reward-hacking signal to watch for.

### `src/train/simpo.py` (Stage 8 — not yet implemented, added last)
- Hand-written, no reference model anywhere — one forward pass per response instead of two, lower VRAM than DPO.
- Reuses `sequence_logprob(..., mean=True)` from `src/model/logprobs.py` rather than reimplementing length normalization.
- `beta` (~2.0–2.5) and `gamma` (~0.5–1.0, or `gamma_beta_ratio` ~0.3–0.5) are more sensitive than DPO's single `beta` — expect a short sweep, don't treat the config defaults as final.
- Explicitly log `mean_len_chosen` / `mean_len_rejected`. Length behaviour is SimPO's whole reason for existing and is the headline of the Stage 9 comparison.

### `src/eval/` (Stage 2 — not yet implemented, build BEFORE training)
- `sycophancy.py` runs the same two-turn pushback protocol in two modes: hold-rate (7a — correct turn-1 answer, wrong pushback) and the stubbornness guard (7b — WRONG turn-1 answer, correct pushback). Both modes must exist before any training starts, so the same harness produces base, DPO, and SimPO numbers with zero methodology drift between them.
- `judge.py` classifies replies (holds / caves / updates-appropriately) via Claude, not string matching. Validate it against ~60 hand labels and keep the agreement number — below ~90% and the downstream hold-rate numbers aren't trustworthy, and the rubric needs work before anything else proceeds.
- `capability.py` is the guardrail against training a stubborn-but-broken model: a GSM8K + MMLU slice, accuracy must stay within ~1–2 points of base.

### `src/data/` (Stage 3 — not yet implemented)
- `generate_pairs.py` sits behind a `PairGenerator` protocol so the Anthropic dependency is swappable, not load-bearing on one provider.
- `build_splits.py` splits by seed id, not by row — a seed's chosen/rejected pair must land entirely in train or entirely in test, never split across both.
- The hand-audit gate (~50 sampled pairs, human sign-off) is a hard stop, not a formality: don't wire training to run off `raw_pairs.jsonl` directly.

### `src/demo/app.py` (Stage 7, extended at Stage 10)
- Gradio side-by-side. Base vs tuned at Stage 7; extended to a base/DPO/SimPO three-way once Stage 9's comparison exists.

## Key design decisions

### Why hand-written losses instead of TRL trainers
The point of this project is demonstrating the mechanics are understood, not that TRL works. TRL is a dependency only inside `tests/`, used once per method to check the hand-written loss against a reference implementation on a toy batch. If a future session is tempted to just call `DPOTrainer` for expediency — don't; that defeats the project's stated purpose (`BUILD_PLAN.md` section 1).

### Why the reference model is "adapter disabled", not a second model
A second full copy of a 3B model in memory is unnecessary VRAM pressure when LoRA already gives us the mechanism for free: same base weights, adapter on vs. off. `peft`'s adapter-disable context manager is the whole trick.

### Why SimPO comes after DPO, not alongside it
`BUILD_PLAN.md` section 1's scope discipline is explicit: a clean DPO project (through Stage 7) is a complete, shippable deliverable on its own. SimPO is strictly additive and shares the same data/eval harness, so building it second means the Stage 9 comparison has exactly one true variable (the objective) instead of two moving parts that were never independently validated.

### Why the eval harness is built before the data pipeline (Stage 2 before Stage 3)
Baseline numbers on the untrained base model have to be captured before the model changes, or there's no "before" in "before/after". Building the eval harness first also forces the judge rubric and the stubbornness-guard cases to be nailed down independent of whatever a trained model happens to produce — it avoids designing the eval around the tuned model's behaviour after the fact.

## Data flow summary
```
seeds.jsonl (400-800 hand-checked Qs)
    |
Stage 3: generate_pairs.py (Claude) -> raw_pairs.jsonl
    |
    v
build_splits.py: dedupe -> hand-audit gate -> split by seed id
    |
    v
train.jsonl / test.jsonl   {prompt, chosen, rejected}
    |
    +--> Stage 4 (optional): sft.py -> SFT LoRA adapter
    |
    +--> Stage 5: dpo.py   (policy = SFT-or-base + LoRA, ref = same model, adapter disabled)
    |         -> DPO LoRA adapter
    |
    +--> Stage 8: simpo.py (policy only, no reference)
              -> SimPO LoRA adapter

test.jsonl + capability benchmark slice
    |
    v
src/eval/{sycophancy,judge,capability}.py, run identically against
base / SFT / DPO / SimPO adapters
    |
    v
results/before_after.md (Stage 6), results/dpo_vs_simpo.md (Stage 9)
```
