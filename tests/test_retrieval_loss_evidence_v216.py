import pytest

from src.xgb_matcher.retrieval_loss_evidence_v216 import geometry, nominal_evidence, summarize


def fact(text):
    return dict(name_text=text, lexical_eligible=True)


def test_split_site_parent_evidence_not_one_alias():
    r = nominal_evidence('COMMUNE ABC ECOLE DEF', [fact('COMMUNE ABC'), fact('ECOLE DEF')])
    assert r['literal_name_overlap'] and not r['whole_query_in_one_official_alias']


def test_extra_crm_context_rejects_conjunction():
    r = nominal_evidence('CLIENT XYZ SITE ABC', [fact('CLIENT XYZ')])
    assert r['literal_name_overlap'] and not r['whole_query_in_one_official_alias']


def test_exact_name():
    assert nominal_evidence('ABC', [fact('ABC XYZ')])['whole_query_in_one_official_alias']


def test_no_literal_not_claim_no_semantic_evidence():
    r = nominal_evidence('AB', [fact('ALPHA BETA')])
    assert not r['literal_name_overlap']
    assert 'semantic_absence' not in r


def test_incompatible_ban_does_not_supply_a_distance():
    assert geometry({}, [0., 0.])['distance_m'] is None


@pytest.mark.parametrize('weak,radius', [(False, 300.), (True, 600.)])
def test_ban_radius_and_no_equivalence(weak, radius):
    r = geometry(dict(spatial_crs_compatible=True, ban_uncertainty_m=200.,
                      ban_declared_weak=weak, ban_x_l93=0., ban_y_l93=0.), [400., 0.])
    assert r['radius_m'] == radius and r['inside_current_radius'] is weak
    assert r['operational_equivalence_inferred'] is False


def test_all21_not_subset():
    with pytest.raises(ValueError): summarize([])
