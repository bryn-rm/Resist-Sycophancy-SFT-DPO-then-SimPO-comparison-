"""Sequence log-prob primitive. Everything in DPO/SimPO rests on this.

Stage 1 (see BUILD_PLAN.md section 4). Implement this and tests/test_logprobs.py
before touching src/train/.
"""
from __future__ import annotations

import torch


def sequence_logprob(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    mean: bool = False,
) -> torch.Tensor:
    """Log-likelihood the model assigns to `response_text` given `prompt_text`.

    1. Tokenize prompt + response with the model's chat template, concatenate.
    2. labels = input_ids with prompt-region tokens set to -100.
    3. Forward pass, shift logits/labels by one position (next-token prediction).
    4. log_softmax + gather the log-prob of each actual response token.
    5. Mask out -100 positions, then sum over response tokens (DPO default),
       or take the mean (`mean=True`, SimPO's length-normalized reward).

    Returns a per-example tensor.
    """
    raise NotImplementedError("Stage 1 - see BUILD_PLAN.md section 4")


def sequence_logprob_batch(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    mean: bool = False,
) -> torch.Tensor:
    """Batched version of sequence_logprob."""
    raise NotImplementedError("Stage 1 - see BUILD_PLAN.md section 4")
