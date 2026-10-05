"""All2000 direct structural initial100, GT CLOSED until every actual exit PASS.

New bounded native proof metric217; own engine closed after EACH query.
No partial successful batch, previous probe or STOP fragment can be evaluated.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from scripts import retrieval_v211_structural_channels as backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, POLICY, quiet, validate_report, parent_exit
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs, check_epochs
from scripts import retrieval_v328_serving_panel as population
from scripts.retrieval_v328_serving_panel import population_rows
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles, identity
from src.xgb_matcher.retrieval_regional_initial_v286 import Evidence, search
from src.xgb_matcher.retrieval_regional_abbreviation_v285 import Regions
from scripts.retrieval_v78_safe_sparse_initial import validate_models
from scripts import retrieval_v228_fit_national_place as training
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT/'config/retrieval_v323_corroborated_population.json'
STATUS = 'SEALED_V323_ALL2000_CORROBORATED_INITIAL100_PILOT'
SAFE_FALSE = ('truth_opened', 'recall_measured', 'old_candidates_opened', 'ready_for_fold1',
              'fold0_opened', 'fold1_opened', 'downstream_models_changed', 'slo_validated')
FALSE = {k: False for k in SAFE_FALSE}


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['resources'] != LIMITS or cfg['queries'] != 2000 or type(cfg['queries']) is not int
            or type(cfg['batch_width']) is not int or cfg['batch_width'] != 4
            or cfg['quiet_seconds'] != 30 or cfg['query_seconds'] != 20 or cfg['mean16_seconds'] is not None
            or verify(cfg['backend_config']).resolve() != backend.CONFIG.resolve()):
        raise ValueError('Frozen whole2000 physical batches, official sources and unchanged guards required')
    return cfg


def implementation():
    refs = backend.implementation()
    refs.update(population.implementation())
    for path in (Path(__file__), CONFIG, ROOT/'docs/retrieval_v322_large_panel_contract.md',
                 ROOT/'docs/retrieval_v328_serving_panel_contract.md',
                 ROOT/'src/xgb_matcher/retrieval_regional_initial_v286.py',
                 ROOT/'src/xgb_matcher/retrieval_regional_abbreviation_v285.py',
                 ROOT/'scripts/retrieval_v281_official_region_snapshots.py',
                 ROOT/'docs/retrieval_v281_official_regions_contract.md',
                 ROOT/'src/xgb_matcher/retrieval_corroborated_initial_v257.py',
                 ROOT/'src/xgb_matcher/retrieval_corroborated_initial_v217.py',
                 ROOT/'src/xgb_matcher/retrieval_nominal_abbreviation_v256.py',
                 ROOT/'src/xgb_matcher/retrieval_single_abbreviation_v246.py',
                 ROOT/'src/xgb_matcher/retrieval_compound_initialisms_v222.py',
                 ROOT/'src/xgb_matcher/retrieval_national_place_v228.py',
                 ROOT/'scripts/retrieval_v228_fit_national_place.py',
                 ROOT/'src/xgb_matcher/retrieval_proof_v56.py',
                 ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',
                 ROOT/'scripts/retrieval_v78_safe_sparse_initial.py',
                 ROOT/'src/xgb_matcher/retrieval_postscore_audit_v187.py',
                 ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',
                 ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py'):
        refs[str(path)] = record(path)
    return refs


def checked_update(models, updated):
    if set(updated) != {'2','3','4'} or set(models) != set(updated):
        raise ValueError('All three safe initial heads required')
    for f, m in updated.items():
        original = models[f]
        if ({k:v for k,v in m.items() if k not in ('weights','place_training')} !=
                {k:v for k,v in original.items() if k != 'weights'}
                or len(m['weights']) != 11
                or any(m['weights'][i] != original['weights'][i] for i in range(11) if i != 4)
                or not math.isfinite(m['weights'][4]) or not 0 <= m['weights'][4] <= 64
                or set(m['place_training']['query_ids']) - set(original['query_ids'])
                or m['place_training']['changed_indices'] != [4]
                or m['place_training']['heldout_entity_overlap'] != 0):
            raise ValueError('Only actual safe trained initial place coefficient may change')
    return updated


def serving(files, queries):
    cfg = config()
    panel, manifest, expansion = population.sealed(Path(cfg['panel_root']))
    if panel['queries'] != list(queries):
        raise ValueError('Same sealed new2000 panel required')
    files.file(manifest['panel']); files.file(manifest['expansions'])
    models = json.loads(files.file(panel['models']).read_text())
    validate_models(models)
    for m in models.values():
        if (set(m['query_ids']) & {q['query_id'] for q in queries}
                or len(m['weights']) != 11 or any(not math.isfinite(w) or not 0 <= w <= 64 for w in m['weights'])
                or not math.isfinite(m['alpha']) or m['alpha'] < 0):
            raise ValueError('Entire TRAIN disjoint panel and original OOF heads required')
        files.file(m['projection']); files.file(m['vocabulary'])
    files.file(cfg['national_place_training'])
    updated, fitted_manifest = training.sealed(Path(cfg['national_place_training']['path']).parent)
    files.file(fitted_manifest['models'])
    models = checked_update(models, updated); validate_models(models)
    for r in expansion['rows']:
        if len(r['terms']) > 8 or any(not isinstance(t, str) or not t.isalpha() or
                not isinstance(w, (float, int)) or not math.isfinite(w) or w <= 0 for t,w in r['terms']):
            raise ValueError('Official words only in same frozen sparse decoder')
    return models, {r['query_id']:r['terms'] for r in expansion['rows']}

def positions(batch):
    if type(batch) is not int or not 0 <= batch < 500:
        raise ValueError('Exact physical batch0..499 required')
    return list(range(4*batch, 4*batch+4))


def load(root):
    root = Path(root)
    cfg = config()
    if (not root.resolve().is_relative_to(Path(cfg['output_root']).resolve())
            or (root/'status.json').exists() or any(root.glob('batch*/worker_stop.json'))
            or any(root.glob('batch*/supervisor_stop.json'))):
        raise ValueError('Own fresh closed-label root, no STOP reuse or retry')
    launch = json.loads((root/'launch.json').read_text())
    if launch['config'] != cfg or any(launch.get(k) is not False for k in SAFE_FALSE):
        raise ValueError('Original protected unchanged generation launch required')
    population_rows(launch['queries'])
    panel, pm, _ = population.sealed(Path(cfg['panel_root']))
    if launch['queries'] != panel['queries'] or launch['panel_manifest'] != record(Path(cfg['panel_root'])/'manifest.json'):
        raise ValueError('Same entire source-bound label-blind2000 population required')
    for ref in launch['implementation'].values():
        verify(ref)
    for ref in (*launch['query_sources'], launch['query_only_launch'], launch['geo_queries'],
                launch['index_manifest'], launch['index_meta']):
        verify(ref)
    if backend.v38.inventory() != launch['inventory']:
        raise ValueError('Original native index inventory changed')
    return launch


def valid_rows(rows, queries, expected):
    population_rows(list(queries))
    if len(rows) != len(expected):
        raise ValueError('Every original expected query output required')
    for result, pos in zip(rows, expected):
        q = queries[pos]
        if (type(result['position']) is not int or result['position'] != pos
                or result['query_id'] != q['query_id'] or type(result['fold']) is not int or result['fold'] != q['fold']
                or not isinstance(result['seconds']['primary'], (float, int)) or isinstance(result['seconds']['primary'], bool)
                or not math.isfinite(result['seconds']['primary']) or not 0 <= result['seconds']['primary'] <= 20
                or len(result['results']['primary']) > 100):
            raise ValueError('Original ordered controls, query budget and direct100 required')
        found = set()
        for rank, candidate in enumerate(result['results']['primary'], 1):
            s, p = identity(candidate)
            if (s in found or type(candidate['retrieval_rank']) is not int or candidate['retrieval_rank'] != rank
                    or candidate['channel'] != 'joint_proof'
                    or isinstance(candidate['source_score'], bool) or not isinstance(candidate['source_score'], (float, int))
                    or not math.isfinite(candidate['source_score']) or candidate['source_score'] < 0):
                raise ValueError('Unique direct output and original emission rank required')
            found.add(s)
        trace = result['primary_trace']
        if (trace['policy'] != 'ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS' or trace['fixed_reservoir'] is not None
                or trace['score_after_fixed_pool'] is not False or trace['initial_cap'] != 100
                or trace['downstream_models_changed'] is not False or trace['label_lookup'] is not False
                or trace['identifiers_as_features'] is not False or trace['recall_measured'] is not False
                or trace['native_leaf_sites'] != 43896818 or trace['full_appended_leaf_universe'] is not False
                or trace['exact_global_fused_score_top100'] is not True
                or any(type(n) is not int or n < 0 for n in trace['source_admitted'].values())
                or sum(trace['source_admitted'].values()) != len(found)):
            raise ValueError('Same frozen direct recipe and honestly limited leaf corpus required')


def stage(root, batch):
    return _stage(root, batch, load(root))


def _stage(root, batch, launch):
    # A full frozen population is loaded by our caller before this loop.
    # Still verify each actual producer/preflight/payload/epoch independently.
    dest = root/f'batch{batch:03d}'
    pos = positions(batch)
    parent_exit(dest)
    gate = json.loads((dest/'preflight.json').read_text())
    if (gate['policy_id'] != POLICY or gate['pass_system_quiet_policy'] is not True
            or gate['worker_swap_budget_changed'] is not True or gate['quiet_seconds'] < 30):
        raise ValueError('Actual own quiet preflight required')
    manifest = json.loads((dest/'manifest.json').read_text())
    validate_report(manifest['resources'])
    if (manifest['status'] != 'QUARANTINED_V323_BATCH' or manifest['launch'] != record(root/'launch.json')
            or manifest['preflight'] != record(dest/'preflight.json') or manifest['positions'] != pos
            or any(manifest.get(k) is not False for k in SAFE_FALSE)):
        raise ValueError('Actual own entire protected pair authority required')
    payload = json.loads(verify(manifest['payload']).read_text())
    if (verify(manifest['payload']).resolve() != (dest/'payload.json').resolve()
            or any(payload.get(k) is not False for k in SAFE_FALSE)
            or payload['used_sources_full_hash_verified_pre_and_post'] is not True
            or payload['native_segments_full_hash_verified'] is not False):
        raise ValueError('Full original used source hashes and native limits required')
    valid_rows(payload['rows'], launch['queries'], pos)
    entries = canonical_epochs(payload['source_epochs'])
    if not entries:
        raise ValueError('Original pre-read source authority cannot be empty')
    check_epochs(entries)
    return manifest, payload


def worker(root, batch):
    launch = load(root)
    backend_cfg, source, old, _, _ = backend.inputs()
    own_cfg = launch['config']
    panel, _, _ = population.sealed(Path(own_cfg['panel_root']))
    queries = population_rows(panel['queries'])
    geos = {g['query_id']:g for g in panel['geo_queries']}
    if list(queries) != launch['queries']:
        raise ValueError('Original complete2000 alignment required')
    pos = positions(batch)
    dest = root/f'batch{batch:03d}'
    if any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Fresh own physical pair only, no existing output overwrite')
    rows, budget, evidence, native = [], None, None, None
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure:
                    raise RuntimeError(budget.failure)
                if time.monotonic()-budget.start > 80:
                    raise TimeoutError('STOP_CORROBORATED_BATCH_WORKER_COST')
            files = SourceFiles(guard)
            evidence = Evidence(source, files)
            models, terms = serving(files, queries)
            regions = Regions(files, own_cfg['regional_source'])
            for p in pos:
                raw = queries[p]
                if any(re.search(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)', raw[k]) for k in ('crm_name','crm_street')):
                    raise ValueError('Identifier-shaped name/street text cannot be a feature; no row omission')
                q = RetrievalQuery(name=normalized(raw['crm_name']),
                    address=normalized(' '.join(raw[k] for k in ('crm_number', 'crm_suffix', 'crm_street'))),
                    number=raw['crm_number'], postcode=raw['crm_postcode'], insee=raw['crm_insee'])
                began = time.monotonic()
                native = TantivyBackend(backend.v38.INDEX)
                if native.searcher.num_docs != launch['index_documents']:
                    raise ValueError('Actual native corpus cardinality drift')
                remaining = 20-(time.monotonic()-began)
                if remaining <= 0:
                    raise TimeoutError('STOP_NATIVE_INDEX_OPEN_COST')
                model = models[str(raw['fold'])]
                found, trace = search(native, evidence, q, geos[raw['query_id']],
                    model['weights'], terms[raw['query_id']], model['alpha'], regions.names, max_seconds=remaining)
                elapsed = time.monotonic()-began
                trace['index_open_counted_in_query_seconds'] = True
                rows.append(dict(query_id=raw['query_id'], fold=raw['fold'], position=p,
                    results=dict(primary=found), seconds=dict(primary=elapsed), primary_trace=trace))
                print(json.dumps(dict(position=p, candidates=len(found), seconds=elapsed,
                                      channels=trace['source_admitted'], geometry=trace['geometry'])), flush=True)
                # Only our completed engine, BEFORE next query or hash. No
                # modification to the metric, index, source or external app.
                native = None
                gc.collect()
                guard()
            regions.close()
            evidence.close()
            evidence = None
            gc.collect()
            files.verify_after()
            load(root)
            valid_rows(rows, queries, pos)
            write_json(dest/'payload.json', dict(rows=rows, source_epochs=list(files.entries.values()),
                used_sources_full_hash_verified_pre_and_post=True, native_segments_full_hash_verified=False, **FALSE))
        validate_report(budget.report())
        write_json(dest/'manifest.json', dict(status='QUARANTINED_V323_BATCH', launch=record(root/'launch.json'),
            positions=pos, preflight=record(dest/'preflight.json'), payload=record(dest/'payload.json'),
            resources=budget.report(), **FALSE))
    except BaseException as exc:
        write_json(dest/'worker_stop.json', dict(status='STOP', reason=str(exc), completed=len(rows),
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None, **FALSE))
        raise
    finally:
        native = None
        if evidence is not None:
            evidence.close()


def admission(root):
    """All16 actual PASS exits,2000 controls and original source epochs; no GT."""
    launch = load(root)
    rows, pairs, epochs, reports = [], [], [], []
    for b in range(500):
        m, p = _stage(root, b, launch)
        rows.extend(p['rows'])
        epochs.extend(p['source_epochs'])
        reports.append(m['resources'])
        pairs.append(record(root/f'batch{b:03d}'/'manifest.json'))
    valid_rows(rows, launch['queries'], list(range(2000)))
    epochs = canonical_epochs(epochs)
    check_epochs(epochs)
    gate = json.loads((root/'cost_gate.json').read_text())
    mean = sum(r['seconds']['primary'] for r in rows[:16])/16
    if (gate != dict(queries=16, mean_seconds=mean, passed=None,
                     truth_opened=False, production_slo_validated=False, eliminatory=False)):
        raise ValueError('Original exact first16 descriptive cost observation required')
    return dict(rows=rows, pairs=pairs, source_epochs=epochs,
                peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in reports),
                worker_seconds=sum(r['wall_seconds'] for r in reports))


def sealed(root):
    """Refuse LIVE/STOP/partial BEFORE any target-side caller can proceed."""
    launch = load(root)
    m = json.loads((root/'manifest.json').read_text())
    if (m.get('status') != STATUS or m.get('queries') != 2000 or type(m.get('queries')) is not int
            or m.get('population') != 2000 or type(m.get('population')) is not int
            or any(m.get(k) is not False for k in SAFE_FALSE)
            or m.get('all2000_outputs_sealed') is not True or m.get('initial100_emitted') is not True
            or m.get('full_appended_leaf_universe') is not False or m['launch'] != record(root/'launch.json')):
        raise ValueError('Actual full2000 protected initial100 root required')
    value = admission(root)
    outputs = json.loads(verify(m['outputs']).read_text())
    if (outputs['rows'] != value['rows'] or outputs['source_epochs'] != value['source_epochs']
            or m['pairs'] != value['pairs'] or m['peak_worker_rss_bytes'] != value['peak_worker_rss_bytes']
            or m['outputs'] != record(root/'outputs.json') or any(outputs.get(k) is not False for k in SAFE_FALSE)):
        raise ValueError('Actual full output and source/producer binding required')
    return launch, m, outputs


def run(root):
    cfg = config()
    if root.exists() or not root.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('Fresh dedicated229 root required')
    _, base, old, _, _ = backend.inputs()
    panel_root = Path(cfg['panel_root'])
    panel, _, _ = population.sealed(panel_root)
    queries = population_rows(panel['queries'])
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        launch = dict(config=cfg, queries=list(queries), implementation=implementation(),
            query_sources=[record(panel_root/'source/panel.json')], query_only_launch=record(panel_root/'population_manifest.json'),
            geo_queries=record(panel_root/'source/panel.json'), panel_manifest=record(panel_root/'manifest.json'),
            inventory=old['inventory'], index_documents=old['index_documents'], index_manifest=old['index_manifest'],
            index_meta=old['index_meta'], backend_config=cfg['backend_config'],
            native_leaf_sites=43896818, full_appended_leaf_universe=False, development_panel_reused=True,
            panel_scope=panel['metadata'], **FALSE)
        write_json(root/'launch.json', launch)
        rows = []
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'RAYON_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        for b in range(500):
            load(root)
            dest = root/f'batch{b:03d}'
            dest.mkdir()
            write_json(dest/'preflight.json', quiet(30, 50))
            with (dest/'worker.log').open('x') as log:
                s = supervise([sys.executable, '-u', '-m', 'scripts.retrieval_v323_corroborated_population',
                    'worker', '--output', str(root), '--batch', str(b)], log, env, dest/'supervisor_stop.json')
            write_json(dest/'supervision.json', s)
            m, p = stage(root, b)
            rows.extend(p['rows'])
            print(json.dumps(dict(generated=len(rows), worker=m['resources'])), flush=True)
            if len(rows) == 16:
                mean = sum(r['seconds']['primary'] for r in rows)/16
                write_json(root/'cost_gate.json', dict(queries=16, mean_seconds=mean, passed=None,
                    truth_opened=False, production_slo_validated=False, eliminatory=False))
        value = admission(root)
        write_json(root/'outputs.json', dict(rows=value['rows'], source_epochs=value['source_epochs'], **FALSE))
        write_json(root/'manifest.json', dict(status=STATUS, queries=2000, population=2000,
            all2000_outputs_sealed=True, initial100_emitted=True, launch=record(root/'launch.json'),
            outputs=record(root/'outputs.json'), pairs=value['pairs'], peak_worker_rss_bytes=value['peak_worker_rss_bytes'],
            worker_seconds=value['worker_seconds'], wall_seconds=time.monotonic()-began,
            native_leaf_sites=43896818, full_appended_leaf_universe=False, native_segments_full_hash_verified=False, **FALSE))
        sealed(root)
        print(json.dumps(dict(status=STATUS, queries=2000, truth_opened=False)), flush=True)
    except BaseException as exc:
        write_json(root/'status.json', dict(status='STOP', reason=str(exc), completed=len(rows) if 'rows' in locals() else 0,
                                          seconds=time.monotonic()-began, **FALSE))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch', type=int)
    args = parser.parse_args()
    if args.mode == 'run':
        run(args.output)
    else:
        worker(args.output, args.batch)
