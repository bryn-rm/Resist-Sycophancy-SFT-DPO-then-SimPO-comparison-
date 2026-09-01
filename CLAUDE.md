# CLAUDE.md - Sycophancy-DPO

Project-specific instructions, in the spirit of Karpathy's CLAUDE.md: short, imperative, principle-first. Full spec and stage acceptance checks: `BUILD_PLAN.md`. Module notes and design rationale: `docs/ARCHITECTURE.md`. Merge those in as needed — don't inline them here. Update the Status line as each stage lands; keep everything else as short as it currently is.

## Status
Stage 1 (log-prob primitive) done and verified: `sequence_logprob`/`sequence_logprob_batch` pass 3/3 offline tests, and a real Qwen2.5-3B spot check scored a holding response far above a caving one (-8.86 vs -40.10 summed logp). Everything past Stage 1 raises `NotImplementedError`. Work stages in order — see the roadmap below and `BUILD_PLAN.md` section 8 for each stage's acceptance check.

## 1. Scope Discipline
**Build exactly what `BUILD_PLAN.md` section 1 specifies. Nothing else.**
- DPO and SimPO only. No PPO, no reward model, no KTO/ORPO/other method.
- Hand-write both losses. `trl` is a test-only dependency (`test_dpo_loss.py`, `test_simpo_loss.py`) — never import it under `src/`.
- LoRA only, no full fine-tuning. Stay in the 3B-instruct class.
- Don't widen the target behaviour beyond sycophancy-under-pushback.
- A shortcut like calling `DPOTrainer` directly defeats the point of this project. Don't take it, even if it's faster.

## 2. Surgical Changes — known landmines
Touch only what the task needs, but get these exactly right when you do:
- **Prompt masking** (`sequence_logprob`) — prompt tokens must be `-100`'d out or log-probs silently include tokens the model didn't generate. Re-verify the mask whenever the tokenizer/chat template changes.
- **Reference model** — the same `PeftModel` with `disable_adapter()`, never a second `from_pretrained`. Under 4-bit, confirm `disable_adapter()` actually recovers base logits before trusting reference log-probs.
- **Seed leakage** — `build_splits.py` splits by seed id, not row. A leak here silently invalidates the "held-out test set" claim in the README.
- **`apply_chat_template(tokenize=True)`** — returns a `BatchEncoding` on the installed `transformers==5.16.1`, not a bare list. Go through `_token_ids()` in `logprobs.py`; don't assume the return type.
- **`eval` as a package name** — shadows the builtin only within `src/eval/`'s own namespace, but avoid `from eval import *` anyway.

## 3. Goal-Driven Execution
Every stage in `BUILD_PLAN.md` section 8 has an explicit acceptance check — that's the definition of done, not the code existing.
- `test_logprobs.py`, `test_dpo_loss.py`, `test_simpo_loss.py` are `pytest.mark.skip`'d stubs. Unskip a test only when its stage's acceptance check passes. Don't delete skip markers preemptively; don't call a stage done with its test still skipped.
- Don't start a stage before the previous one's acceptance check is verifiably true.

```
0 Scaffold ✓   1 Log-probs ✓   2 Eval harness   3 Data   4 SFT (optional)
5 DPO   6 Evaluate   7 Demo+README (DPO ships alone)   8 SimPO   9 Compare   10 Writeup
```
