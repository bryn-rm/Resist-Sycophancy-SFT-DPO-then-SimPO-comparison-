"""Sequence log-prob primitive. Everything in DPO/SimPO rests on this.

Stage 1 (see BUILD_PLAN.md section 4). Implement this and tests/test_logprobs.py
before touching src/train/.
"""
from __future__ import annotations

import torch

IGNORE_INDEX = -100


def _token_ids(result) -> list[int]:
    """Normalize a tokenizer call's return value to a plain list[int].

    `apply_chat_template(..., tokenize=True)` returns a bare list[int] on
    older transformers but a dict-like `BatchEncoding` (from which the ids
    live under "input_ids") on newer ones (verified against the installed
    transformers 5.16.1 / Qwen2.5-3B-Instruct's real chat template - the
    bare-list assumption silently iterated over dict keys instead of token
    ids). Handle both.
    """
    if isinstance(result, list):
        return result
    return list(result["input_ids"])


def _encode_prompt_and_response(tokenizer, prompt_text: str, response_text: str) -> tuple[list[int], list[int]]:
    """Encode prompt and response as two separate token-id lists.

    The prompt is rendered through the chat template (as a single user turn,
    with the assistant generation prompt appended); the response is encoded
    as a plain continuation and concatenated. Encoding them separately -
    rather than rendering the full templated conversation as one string and
    re-tokenizing - means the prompt/response boundary is known exactly by
    construction, instead of having to be recovered after the fact. See the
    "chat template drift" gotcha in CLAUDE.md.
    """
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    response_ids = _token_ids(tokenizer(response_text, add_special_tokens=False))
    if tokenizer.eos_token_id is not None and (
        len(response_ids) == 0 or response_ids[-1] != tokenizer.eos_token_id
    ):
        response_ids = list(response_ids) + [tokenizer.eos_token_id]
    return list(prompt_ids), response_ids


def _gather_token_logps(logits: torch.Tensor, labels: torch.Tensor, mean: bool) -> torch.Tensor:
    """Shift logits/labels, gather per-token log-probs at IGNORE_INDEX-masked labels, reduce."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels != IGNORE_INDEX
    safe_labels = shift_labels.clamp(min=0)

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask

    summed = token_logps.sum(dim=-1)
    if not mean:
        return summed
    lengths = mask.sum(dim=-1).clamp(min=1)
    return summed / lengths


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
    return sequence_logprob_batch(model, tokenizer, [prompt_text], [response_text], mean=mean)[0]


def sequence_logprob_batch(
    model,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    mean: bool = False,
) -> torch.Tensor:
    """Batched version of sequence_logprob."""
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    all_input_ids = []
    all_labels = []
    for prompt_text, response_text in zip(prompts, responses):
        prompt_ids, response_ids = _encode_prompt_and_response(tokenizer, prompt_text, response_text)
        all_input_ids.append(prompt_ids + response_ids)
        all_labels.append([IGNORE_INDEX] * len(prompt_ids) + response_ids)

    max_len = max(len(ids) for ids in all_input_ids)
    batch_size = len(all_input_ids)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, labs) in enumerate(zip(all_input_ids, all_labels)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(labs)] = torch.tensor(labs, dtype=torch.long)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return _gather_token_logps(outputs.logits, labels, mean=mean)
