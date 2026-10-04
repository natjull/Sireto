"""Quality-oriented new initial100 recipe; GT only AFTER complete128 seal.

Probe24 is cost-only and NEVER authorizes its partial outputs for evaluation.
No failed V206 fragments, trained heads, downstream models or targets read.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, quiet, validate_report
from src.xgb_matcher.retrieval_query_population_v192 import rows as population_rows
from src.xgb_matcher.retrieval_structural_channels_v210 import OfficialEvidence, SourceFiles, search
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v210_structural_channels.json'
STATUS = 'SEALED_V210_DIRECT_STRUCTURAL_CHANNELS_INITIAL100_PILOT'


def inputs():
    cfg = json.loads(CONFIG.read_text())
    if cfg['resources'] != LIMITS or cfg['quiet_seconds'] != 30 or cfg['query_seconds'] != 20 or cfg['mean16_seconds'] != 4:
        raise ValueError('Unchanged authorized resource and experimental query budgets required')
    base = json.loads(verify(cfg['source_config']).read_text())
    old = json.loads(verify(base['query_only_launch']).read_text())
    if old['truth_opened'] or old['old_candidates_opened']:
        raise ValueError('Original full query-only128 authority required')
    rows = population_rows(old['queries'])
    geo = json.loads(verify(old['geo_queries']).read_text())
    if len(geo) != 128 or len({g['query_id'] for g in geo}) != 128 or {g['query_id'] for g in geo} != {q['query_id'] for q in rows}:
        raise ValueError('Original full128 query geometry alignment required')
    for ref in old['query_sources']:
        verify(ref)
    for key in ('index_manifest', 'index_meta'):
        verify(old[key])
    if old['inventory'] != v38.inventory():
        raise ValueError('Original full native index inventory drift')
    return cfg, base, old, rows, {g['query_id']: g for g in geo}


def implementation():
    names = ('retrieval_structural_channels_v210', 'hierarchical_retrieval', 'retrieval_joint_v46',
             'retrieval_name_pages_v151', 'retrieval_national_names_v138', 'retrieval_variants_v118',
             'retrieval_query_population_v192', 'retrieval_memory_policy_v153', 'retrieval_supervisor_v164',
             'retrieval_address_authority_v168', 'retrieval_v20_workbench', 'retrieval_parent_discovery_v18')
    paths = [Path(__file__), CONFIG, ROOT/'docs/retrieval_v210_structural_channels_contract.md',
             ROOT/'scripts/retrieval_v38_initial_top100.py',
             *[ROOT/('src/xgb_matcher/'+n+'.py') for n in names]]
    return {str(p): record(p) for p in paths}


def worker(output, positions):
    launch = json.loads((output/'launch.json').read_text())
    cfg, base, old, queries, geo = inputs()
    if (launch['config'] != cfg or launch['queries'] != list(queries) or launch['inventory'] != v38.inventory()
            or (output/'status.json').exists() or positions != launch['positions']):
        raise ValueError('Fresh unchanged full128 launch; STOP cannot be retried')
    for ref in launch['implementation'].values():
        verify(ref)
    for key in ('index_manifest', 'index_meta'):
        verify(launch[key])
    rows, evidence, budget = [], None, None
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure:
                    raise RuntimeError(budget.failure)
                if time.monotonic()-budget.start > LIMITS['max_seconds']:
                    raise TimeoutError('STOP_STRUCTURAL_CHANNEL_WORKER_COST')
            files = SourceFiles(guard)
            evidence = OfficialEvidence(base, files)
            backend = TantivyBackend(v38.INDEX)
            if backend.searcher.num_docs != launch['index_documents']:
                raise ValueError('Unchanged national native document universe required')
            for pos in positions:
                raw = queries[pos]
                q = RetrievalQuery(name=normalized(raw['crm_name']),
                                   address=normalized(' '.join(raw[k] for k in ('crm_number', 'crm_suffix', 'crm_street'))),
                                   number=raw['crm_number'], postcode=raw['crm_postcode'], insee=raw['crm_insee'])
                found, trace = search(backend, evidence, q, geo[raw['query_id']], max_seconds=cfg['query_seconds'])
                rows.append(dict(query_id=raw['query_id'], fold=raw['fold'], position=pos,
                                 results=dict(primary=found), seconds=dict(primary=trace['seconds']), primary_trace=trace))
                print(json.dumps(dict(position=pos, candidates=len(found), trace=trace)), flush=True)
            files.verify_after()
            evidence.close()
            evidence = None
            if v38.inventory() != launch['inventory']:
                raise ValueError('Original native index inventory drift')
            for key in ('index_manifest', 'index_meta'):
                verify(launch[key])
        validate_report(budget.report())
        write_json(output/'batch.json', dict(rows=rows, resources=budget.report(),
                                            source_epochs=list(files.entries.values()),
                                            original_used_sources_full_hash_verified=True,
                                            native_segments_full_hash_verified=False,
                                            truth_opened=False, recall_measured=False))
    except BaseException as exc:
        write_json(output/'worker_stop.json', dict(status='STOP', reason=str(exc), completed=len(rows),
                                                   truth_opened=False, recall_measured=False))
        raise
    finally:
        if evidence is not None:
            evidence.close()


def probe(output):
    cfg, base, old, queries, geo = inputs()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('Dedicated CATNAT experiment root required')
    output.mkdir(parents=True, exist_ok=False)
    try:
        # Q24 was a known COST failure, not selected using a target/miss.
        launch = dict(config=cfg, queries=list(queries), positions=[24], implementation=implementation(),
                      query_only_launch=base['query_only_launch'], geo_queries=old['geo_queries'],
                      inventory=v38.inventory(), index_documents=old['index_documents'],
                      index_manifest=old['index_manifest'], index_meta=old['index_meta'],
                      truth_opened=False, old_candidates_opened=False, recall_measured=False)
        write_json(output/'launch.json', launch)
        write_json(output/'preflight.json', quiet(30, 50))
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'RAYON_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        with (output/'worker.log').open('x') as log:
            supervision = supervise([sys.executable, '-u', '-m', 'scripts.retrieval_v210_structural_channels',
                                     'worker', '--output', str(output)], log, env, output/'supervisor_stop.json')
        write_json(output/'supervision.json', supervision)
        parent_exit(output)
        batch = json.loads((output/'batch.json').read_text())
        validate_report(batch['resources'])
        if (len(batch['rows']) != 1 or batch['rows'][0]['position'] != 24 or batch['truth_opened']
                or len(batch['rows'][0]['results']['primary']) > 100):
            raise ValueError('Complete own probe scope and direct cap100 required')
        write_json(output/'manifest.json', dict(status='SEALED_V210_Q24_COST_ONLY_NOT_RECALL',
                                               launch=record(output/'launch.json'), batch=record(output/'batch.json'),
                                               supervision=record(output/'supervision.json'),
                                               full128_outputs_sealed=False, evaluation_authorized=False,
                                               truth_opened=False, recall_measured=False, ready_for_fold1=False))
        print(json.dumps(dict(status='COST_ONLY_PASS', resources=batch['resources'],
                              trace=batch['rows'][0]['primary_trace'])), flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), truth_opened=False, recall_measured=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe', 'worker'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'probe':
        probe(args.output)
    else:
        worker(args.output, [24])
