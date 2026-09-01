from types import SimpleNamespace

import torch

from model.logprobs import sequence_logprob

VOCAB_SIZE = 32


class TinyCausalLM(torch.nn.Module):
    """Deterministic per-token embedding + linear head, no cross-token mixing.

    Real causal LMs mix information across positions via attention, but this
    primitive's job (shifting, masking, gathering, reducing) doesn't depend
    on that - it only needs *some* fixed, differentiable-shaped logits tensor
    to slice. Keeping the fake model position-independent makes the expected
    values easy to compute by hand in the tests below.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, hidden: int = 8, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab_size, hidden, generator=generator))
            self.head.weight.copy_(torch.randn(vocab_size, hidden, generator=generator))
            self.head.bias.copy_(torch.randn(vocab_size, generator=generator))

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


class FakeTokenizer:
    """Minimal stand-in exposing the subset of the tokenizer API this
    primitive uses: apply_chat_template, __call__, pad/eos ids. Deterministic
    char->id mapping, no network or real chat template needed."""

    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [(ord(c) % (VOCAB_SIZE - 3)) + 3 for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        ids = self._encode(messages[0]["content"])
        if add_generation_prompt:
            ids = ids + [2]
        return ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode(text)}


def _expected_prompt_response_ids(tokenizer, prompt_text, response_text):
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=True, add_generation_prompt=True
    )
    response_ids = tokenizer(response_text)["input_ids"]
    if response_ids[-1] != tokenizer.eos_token_id:
        response_ids = response_ids + [tokenizer.eos_token_id]
    return prompt_ids, response_ids


def test_sequence_logprob_matches_manual_computation():
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    prompt_text, response_text = "hi there", "ok sure"

    result = sequence_logprob(model, tokenizer, prompt_text, response_text)

    prompt_ids, response_ids = _expected_prompt_response_ids(tokenizer, prompt_text, response_text)
    full_ids = torch.tensor([prompt_ids + response_ids])
    with torch.no_grad():
        logits = model(full_ids).logits
    log_probs = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)

    expected = 0.0
    for i, token_id in enumerate(response_ids):
        pos = len(prompt_ids) + i - 1  # position in the shifted logits that predicts this token
        expected += log_probs[pos, token_id].item()

    assert torch.allclose(result, torch.tensor(expected), atol=1e-5)


def test_prompt_tokens_masked():
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    prompt_text, response_text = "a longer prompt here", "short reply"

    result = sequence_logprob(model, tokenizer, prompt_text, response_text)

    prompt_ids, response_ids = _expected_prompt_response_ids(tokenizer, prompt_text, response_text)
    full_ids = torch.tensor([prompt_ids + response_ids])
    with torch.no_grad():
        logits = model(full_ids).logits
    shift_logits = logits[0, :-1, :]
    shift_labels = full_ids[0, 1:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    per_position_logp = log_probs[torch.arange(len(shift_labels)), shift_labels]

    response_only_sum = per_position_logp[len(prompt_ids) - 1 :].sum()
    everything_sum = per_position_logp.sum()

    # Sanity check the fixture: including prompt positions must actually
    # change the sum, or this test can't distinguish masked from unmasked.
    assert not torch.allclose(response_only_sum, everything_sum)

    assert torch.allclose(result, response_only_sum, atol=1e-5)
    assert not torch.allclose(result, everything_sum, atol=1e-5)


def test_mean_mode_matches_sum_divided_by_length():
    model = TinyCausalLM()
    tokenizer = FakeTokenizer()
    prompt_text, response_text = "does this hold", "yes it does"

    summed = sequence_logprob(model, tokenizer, prompt_text, response_text, mean=False)
    mean = sequence_logprob(model, tokenizer, prompt_text, response_text, mean=True)

    _, response_ids = _expected_prompt_response_ids(tokenizer, prompt_text, response_text)
    assert torch.allclose(mean * len(response_ids), summed, atol=1e-5)
