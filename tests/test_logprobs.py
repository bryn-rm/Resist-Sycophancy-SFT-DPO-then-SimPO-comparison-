import pytest


@pytest.mark.skip(reason="Stage 1 not yet implemented - see BUILD_PLAN.md section 4")
def test_sequence_logprob_matches_manual_computation():
    """On a tiny fixed input, the gather-and-sum log-prob must equal an
    independently computed value."""


@pytest.mark.skip(reason="Stage 1 not yet implemented - see BUILD_PLAN.md section 4")
def test_prompt_tokens_masked():
    """Prompt-region tokens must be excluded from the summed log-prob."""


@pytest.mark.skip(reason="Stage 1 not yet implemented - see BUILD_PLAN.md section 4")
def test_mean_mode_matches_sum_divided_by_length():
    """sequence_logprob(..., mean=True) must equal sum-mode divided by the
    response token count (used by SimPO, Stage 8)."""
