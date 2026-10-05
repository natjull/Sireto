"""All128 query-only Data ES fact availability; no candidates, scores or GT."""
import argparse
import json
import os
from pathlib import Path
import sys
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_data_es_access_v309 import Facilities
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_query_population_v192 import rows as population_rows
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, POLICY, quiet, validate_report, parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs, check_epochs
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json
from scripts import retrieval_v287_corroborated_population as baseline

OUTPUT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v319_data_es_availability')
SOURCE = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v308_data_es_source/compile_20261005_a/manifest.json',
    sha256='e6ce75040a24a4567a09b5d79e62ecb14b9c3d0f6401d6465e651033020310a9', size_bytes=1317)
QUERIES = dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v78_safe_sparse_initial/pilot_20261003_a/launch.json',
    sha256='6fc2cb89b489e8038e6f211e0fac0a36954874e066e8a727558cfa01d9d90956', size_bytes=85950)
STATUS = 'SEALED_V319_ALL128_QUERY_ONLY_DATA_ES_FACT_AVAILABILITY'
FALSE = {k: False for k in ('truth_opened', 'recall_measured', 'candidates_emitted', 'candidate_score_computed',
    'fold0_opened', 'fold1_opened', 'old_candidates_opened', 'ready_for_fold1', 'model_fit',
    'downstream_models_changed', 'crm_request_sent', 'slo_validated')}


def query_rows(launch):
    if launch['truth_opened'] is not False or launch['old_candidates_opened'] is not False:
        raise ValueError('Original frozen CRM query-only launch required')
    queries = launch['queries']
    population_rows(queries)
    return queries


def summarize(rows):
    if len(rows) != 128 or [r['position'] for r in rows] != list(range(128)):
        raise ValueError('Complete ordered128 availability, never partial population')
    return dict(queries=128, queries_with_explicit_endpoints=sum(r['facts']['distinct_explicit_endpoints'] > 0 for r in rows),
        queries_with_source_name_observations=sum(r['facts']['matched_observations'] > 0 for r in rows),
        query_endpoint_incidences=sum(r['facts']['distinct_explicit_endpoints'] for r in rows),
        matched_observation_incidences=sum(r['facts']['matched_observations'] for r in rows),
        queries_with_unknown_identifier_records=sum(r['facts']['source_records_missing_explicit_siret'] > 0 for r in rows),
        locality_incidences={k: sum(r['facts']['locality_counts'][k] for r in rows) for k in ('local', 'unknown', 'nonlocal')})


def implementation():
    refs = baseline.implementation()
    for path in (Path(__file__), ROOT/'docs/retrieval_v319_data_es_availability_contract.md',
        ROOT/'src/xgb_matcher/retrieval_data_es_access_v309.py', ROOT/'docs/retrieval_v309_data_es_access_contract.md',
        ROOT/'scripts/retrieval_v308_data_es_source.py', ROOT/'docs/retrieval_v308_data_es_source_contract.md',
        ROOT/'src/xgb_matcher/retrieval_contained_parent_v300.py'):
        refs[str(path)] = record(path)
    return refs


def load(root):
    root = Path(root)
    if not root.resolve().is_relative_to(OUTPUT.resolve()) or (root/'status.json').exists() or any(root.glob('phase/*stop.json')):
        raise ValueError('Fresh own query-only root, no STOP reuse')
    l = json.loads((root/'launch.json').read_text())
    if l['source'] != SOURCE or l['query_launch'] != QUERIES or l['limits'] != LIMITS or any(l.get(k) is not False for k in FALSE):
        raise ValueError('Same immutable all128 query-only launch required')
    for ref in l['implementation'].values():
        verify(ref)
    return l


def worker(root):
    load(root)
    d = root/'phase'
    facilities = None
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure:
                    raise RuntimeError(budget.failure)
            files = SourceFiles(guard)
            queries = query_rows(json.loads(files.file(QUERIES).read_text()))
            facilities = Facilities(files, SOURCE)
            rows = []
            for position, raw in enumerate(queries):
                guard()
                q = RetrievalQuery(name=normalized(raw['crm_name']), insee=raw['crm_insee'], postcode=raw['crm_postcode'])
                _, trace = facilities.facts(q)
                rows.append(dict(position=position, query_id=raw['query_id'], fold=raw['fold'], facts=trace))
            facilities.close()
            facilities = None
            files.verify_after()
            load(root)
            write_json(d/'payload.json', dict(rows=rows, summary=summarize(rows),
                source_epochs=list(files.entries.values()), all128_before_analysis=True, query_only=True, **FALSE))
        validate_report(budget.report())
        write_json(d/'manifest.json', dict(status='QUARANTINED_V319_QUERY_ONLY', launch=record(root/'launch.json'),
            preflight=record(d/'preflight.json'), payload=record(d/'payload.json'), resources=budget.report(), **FALSE))
    except BaseException as exc:
        write_json(d/'worker_stop.json', dict(status='STOP', reason=str(exc), **FALSE))
        raise
    finally:
        if facilities is not None:
            facilities.close()


def admission(root):
    root = Path(root)
    load(root)
    d = root/'phase'
    parent_exit(d)
    m = json.loads((d/'manifest.json').read_text())
    validate_report(m['resources'])
    gate = json.loads(verify(m['preflight']).read_text())
    p = json.loads(verify(m['payload']).read_text())
    if (m['status'] != 'QUARANTINED_V319_QUERY_ONLY' or m['launch'] != record(root/'launch.json')
        or m['preflight'] != record(d/'preflight.json') or m['payload'] != record(d/'payload.json')
        or gate['policy_id'] != POLICY or gate['quiet_seconds'] < 30 or gate['pass_system_quiet_policy'] is not True
        or p['query_only'] is not True or p['all128_before_analysis'] is not True
        or any(x.get(k) is not False for x in (m, p) for k in FALSE)):
        raise ValueError('Actual own successful complete query-only producer required')
    queries = query_rows(json.loads(verify(QUERIES).read_text()))
    if ([(r['position'], r['query_id'], r['fold']) for r in p['rows']] != [(i, q['query_id'], q['fold']) for i, q in enumerate(queries)]
        or p['summary'] != summarize(p['rows'])):
        raise ValueError('All original128 controls and exact descriptive summary required')
    check_epochs(canonical_epochs(p['source_epochs']))
    return m, p


def sealed(root):
    root = Path(root)
    m, p = admission(root)
    s = json.loads((root/'manifest.json').read_text())
    if (s['status'] != STATUS or s['producer'] != record(root/'phase/manifest.json')
        or s['payload'] != m['payload'] or s['summary'] != p['summary'] or any(s.get(k) is not False for k in FALSE)):
        raise ValueError('Same actual all128 query-only availability required')
    return s, p


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Fresh own query-only output required')
    root.mkdir(parents=True, exist_ok=False)
    d = root/'phase'
    d.mkdir()
    try:
        write_json(root/'launch.json', dict(source=SOURCE, query_launch=QUERIES, limits=LIMITS, implementation=implementation(), **FALSE))
        write_json(d/'preflight.json', quiet(30, 50))
        with (d/'worker.log').open('x') as log:
            supervision = supervise([sys.executable, '-u', '-m', 'scripts.retrieval_v319_data_es_availability', 'worker', '--output', str(root)],
                log, {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, d/'supervisor_stop.json')
        write_json(d/'supervision.json', supervision)
        m, p = admission(root)
        write_json(root/'manifest.json', dict(status=STATUS, producer=record(d/'manifest.json'), payload=m['payload'], summary=p['summary'], **FALSE))
        sealed(root)
        print(json.dumps(dict(status=STATUS, summary=p['summary'])), flush=True)
    except BaseException as exc:
        write_json(root/'status.json', dict(status='STOP', reason=str(exc), **FALSE))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run', 'worker'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    (run if a.mode == 'run' else worker)(a.output)
