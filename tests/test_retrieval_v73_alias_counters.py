from dataclasses import replace

from scripts.audit_retrieval_v73_alias_counters import corrected_counts
from src.xgb_matcher.retrieval_observation_v72 import score, query
from tests.test_retrieval_observation_v72 import observation


def test_duplicate_removal_is_not_a_negative_alias_addition_or_a_receiving_site():
    o = observation('12345678900001')
    o = replace(o, names=(o.names[0], o.names[0]))
    new = replace(o, names=(o.names[0],))
    counts, bindings, removed = corrected_counts([o], [new], [dict(site_evidence_id=o.evidence_id,
        site_source_record_id=o.source_record_id, legal_aliases_added=-1)],
        dict(added_legal_aliases=-1, sites_receiving_legal_alias=1))
    assert counts == dict(added_legal_aliases=0, sites_receiving_legal_alias=0)
    assert bindings[0]['legal_aliases_added'] == 0 and removed == 1
    assert score(o, query(dict(crm_name='ALPHA'))) == score(new, query(dict(crm_name='ALPHA')))
