from scripts import build_synthetic_gt_family_evidence as evidence


def test_bounded_substitutions_requires_equal_length_and_small_observed_edit():
    assert evidence.bounded_substitutions("SOCIETE", "S0CIETE") == [("o", "0")]
    assert evidence.bounded_substitutions("SOCIETE", "SOCIET") == []
    assert evidence.bounded_substitutions("SOCIETE", "SAXIETE") == []


def test_same_tokens_different_order_is_multiset_strict():
    assert evidence.same_tokens_different_order("ALPHA BETA", "BETA ALPHA")
    assert not evidence.same_tokens_different_order("ALPHA BETA", "ALPHA BETA")
    assert not evidence.same_tokens_different_order("ALPHA BETA", "BETA GAMMA")
