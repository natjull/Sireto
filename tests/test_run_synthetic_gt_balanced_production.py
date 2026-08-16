import pytest

from scripts.run_synthetic_gt_balanced_production import next_attempted_variants


def test_next_attempted_variants_uses_full_and_exact_residual_batches() -> None:
    assert next_attempted_variants(3_723, 20_000, 200) == 600
    assert next_attempted_variants(19_873, 20_000, 200) == 127
    assert next_attempted_variants(19_999, 20_000, 200) == 1


def test_next_attempted_variants_rejects_completed_or_invalid_counts() -> None:
    with pytest.raises(ValueError):
        next_attempted_variants(20_000, 20_000, 200)
    with pytest.raises(ValueError):
        next_attempted_variants(10, 20_000, 0)
