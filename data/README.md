# data/

Populated by the Stage 3 pipeline (`src/data/generate_pairs.py`, then
`src/data/build_splits.py`). Nothing is checked in here yet.

- `seeds.jsonl` — `{id, question, gold_answer, domain}`, 400-800 hand-checked,
  unambiguous questions across arithmetic / factual / elementary-logic domains.
- `raw_pairs.jsonl` — generated two-turn chosen/rejected pairs, pre-audit.
- `train.jsonl` — `{prompt, chosen, rejected}`, post-audit, post-split.
- `test.jsonl` — held-out `{prompt, chosen, rejected}`, never used in training.

Splits are by seed id, so no seed appears in both `train.jsonl` and
`test.jsonl`. See `BUILD_PLAN.md` section 6.
