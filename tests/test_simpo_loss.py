import pytest


@pytest.mark.skip(reason="Stage 8 not yet implemented - see BUILD_PLAN.md section 5b")
def test_simpo_loss_matches_trl_cpo_trainer():
    """Confirm the hand-written SimPO loss matches TRL's CPOTrainer
    (loss_type="simpo") on one toy batch, to a tight tolerance."""
