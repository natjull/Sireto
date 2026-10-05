import inspect
import pytest
from scripts import retrieval_v319_data_es_availability as probe


def rows():
    return [dict(position=i, facts=dict(distinct_explicit_endpoints=0, matched_observations=0,
        source_records_missing_explicit_siret=0, locality_counts={k: 0 for k in ('local', 'unknown', 'nonlocal')})) for i in range(128)]


def test_complete_zero_availability_is_not_recall():
    assert probe.summarize(rows()) == dict(queries=128, queries_with_explicit_endpoints=0,
        queries_with_source_name_observations=0, query_endpoint_incidences=0, matched_observation_incidences=0,
        queries_with_unknown_identifier_records=0, locality_incidences={'local': 0, 'unknown': 0, 'nonlocal': 0})
    with pytest.raises(ValueError):
        probe.summarize(rows()[:-1])
    reversed_rows = list(reversed(rows()))
    with pytest.raises(ValueError):
        probe.summarize(reversed_rows)


def test_original_safe_query_scope_only():
    queries = [dict(query_id='q'+str(i), fold=2+i%3, crm_name='NAME', crm_number='',
        crm_suffix='', crm_street='', crm_postcode='', crm_insee='') for i in range(128)]
    launch = dict(truth_opened=False, old_candidates_opened=False, queries=queries)
    assert probe.query_rows(launch) is queries
    queries[-1]['fold'] = 1
    with pytest.raises(ValueError):
        probe.query_rows(launch)
    code = inspect.getsource(probe)
    assert 'human_ground_truth' not in code and 'read_table' not in code and 'evaluate(' not in code
    assert all(v is False for v in probe.FALSE.values())


def test_probe_binds_admitted_data_es_source_not_culture_or_stopped303():
    from src.xgb_matcher import retrieval_data_es_access_v309 as access
    from scripts import retrieval_v308_data_es_source as source
    assert probe.Facilities is access.Facilities
    assert access.source is source
    assert 'retrieval_v308_data_es_source/compile_20261005_a' in probe.SOURCE['path']
    assert probe.SOURCE['sha256'] == 'e6ce75040a24a4567a09b5d79e62ecb14b9c3d0f6401d6465e651033020310a9'
    refs = probe.implementation()
    assert any(p.endswith('/retrieval_data_es_access_v309.py') for p in refs)
    assert not any(p.endswith('/retrieval_culture_access_v311.py') for p in refs)
