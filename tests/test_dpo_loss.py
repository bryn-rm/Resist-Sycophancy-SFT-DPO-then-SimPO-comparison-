import pytest


@pytest.mark.skip(reason="Stage 5 not yet implemented - see BUILD_PLAN.md section 5")
def test_dpo_loss_matches_trl():
    """On one toy batch, the hand-written loss must match trl's DPO loss to
    a tight tolerance. trl is a dev-only dependency, imported only here and
    in test_simpo_loss.py -- never in src/."""
