"""Freeze a label-blind 2000-query panel, then same local frozen sparse encoder.

No GT/old candidate source is read. All original CRM/BAN rows are checked.
No STOP retries. Every producer has actual supervised exit0/PASS before seal.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v287_corroborated_population as baseline
from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v78_safe_sparse_initial import validate_models, expansion_rows
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, POLICY, quiet, validate_report, parent_exit
from src.xgb_matcher.retrieval_postscore_audit_v187 import canonical_epochs, check_epochs
from src.xgb_matcher.retrieval_query_population_v192 import FIELDS, rows as old_rows
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

OUTPUT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v322_large_panel')
SIZE = 2000
SEED = 'initial100-v322-20261005-first-large-panel'
STATUS = 'SEALED_V322_LABEL_BLIND_2000_QUERY_ONLY_PANEL'
ENCODED = 'SEALED_V322_ALL2000_FROZEN_SPARSE_EXPANSIONS'
SAFE_FALSE = baseline.SAFE_FALSE
FALSE = baseline.FALSE
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'RAYON_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}


def population_rows(value, expected=SIZE):
    if (not isinstance(value, list) or len(value) != expected
            or len({q['query_id'] for q in value}) != expected
            or any(set(q) != FIELDS or type(q['fold']) is not int or q['fold'] not in (2, 3, 4)
                   or not q['query_id'] or any(not isinstance(q[k], str) for k in FIELDS-{'fold'}) for q in value)):
        raise ValueError('Entire typed unique query-only CRM population and safe folds required')
    return tuple(q.copy() for q in value)


def priority(q):
    return hashlib.sha256((SEED+'|'+q['query_id']).encode()).hexdigest(), q['query_id']


def choose(all_queries, pilot, train, count=SIZE):
    population_rows(all_queries, 23571)
    old_rows(pilot)
    training = set(train); consumed = {q['query_id'] for q in pilot}
    if (type(count) is not int or count != SIZE or len(training) != 512
            or training & consumed or not training | consumed <= {q['query_id'] for q in all_queries}):
        raise ValueError('All512 TRAIN plus disjoint consumed128 before selection required')
    eligible = [q for q in all_queries if q['query_id'] not in training | consumed]
    if len(eligible) != 22931:
        raise ValueError('Exact full eligible denominator required, no quality filter')
    return sorted(eligible, key=priority)[:count]


def geometry(q, g):
    if (g['query_id'], g['fold']) != (q['query_id'], q['fold']):
        raise ValueError('Original CRM/BAN control alignment required')
    out = g.copy()
    out['spatial_crs_compatible'] = bool(q['crm_insee'] and not q['crm_insee'].startswith(('97', '98'))
        and g['ban_resolved'] and 0 <= g['ban_x_l93'] <= 1300000 and 6000000 <= g['ban_y_l93'] <= 7300000)
    return out


def implementation():
    paths = (Path(__file__), ROOT/'docs/retrieval_v322_large_panel_contract.md',
        ROOT/'src/xgb_matcher/retrieval_contrastive_v52.py', ROOT/'src/xgb_matcher/semantic.py',
        ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py', ROOT/'scripts/retrieval_v78_safe_sparse_initial.py',
        ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py', ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py')
    refs = baseline.implementation()
    refs.update({str(p): record(p) for p in paths})
    return refs


def load(root):
    root = Path(root)
    if (not root.resolve().is_relative_to(OUTPUT) or (root/'status.json').exists()
            or any(root.glob('*/worker_stop.json')) or any(root.glob('*/supervisor_stop.json'))):
        raise ValueError('Fresh dedicated source root, no STOP/partial reuse')
    launch = json.loads((root/'launch.json').read_text())
    if launch['seed'] != SEED or launch['queries'] != SIZE or any(launch.get(k) is not False for k in SAFE_FALSE):
        raise ValueError('Original frozen label-blind sampling authority required')
    for ref in launch['implementation'].values(): verify(ref)
    return launch


def read_sources(files):
    generation = json.loads(files.file(dict(path=str(v38.GEN), sha256=v38.GEN_SHA)).read_text())
    if generation['status'] != 'SEALED_FULL_PAIRED_GENERATION' or generation['queries'] != 23571:
        raise ValueError('Entire original V34 generation source required')
    queries, geos = [], {}
    for ref in generation['batches']:
        b = json.loads(files.file(ref).read_text())
        if b['start'] != len(queries) or not 0 < b['count'] <= 32:
            raise ValueError('Complete contiguous original CRM batches required')
        before = len(queries)
        for sr in b['shards']:
            s = json.loads(files.file(sr).read_text())
            qr, gr = (s['files'][k] for k in ('queries.parquet', 'ban.parquet'))
            qp, gp = files.file(qr), files.file(gr)
            if set(pq.read_schema(qp).names) != FIELDS:
                raise ValueError('No labels/candidates allowed in query source')
            part = pq.read_table(qp, filters=[('fold','in',[2,3,4])], use_threads=False).to_pylist()
            gs = pq.read_table(gp, filters=[('fold','in',[2,3,4])], use_threads=False).to_pylist()
            if len(part) != pq.read_metadata(qp).num_rows or len(gs) != pq.read_metadata(gp).num_rows:
                raise ValueError('Closed folds must never be read or silently filtered from source')
            if len(part) != len(gs): raise ValueError('BAN cannot remove queries')
            by_id = {g['query_id']: g for g in gs}
            if len(by_id) != len(gs): raise ValueError('Duplicate BAN controls')
            for q in part:
                if q['query_id'] in geos: raise ValueError('Duplicate original control')
                geos[q['query_id']] = geometry(q, by_id[q['query_id']])
            queries.extend(part)
        if len(queries)-before != b['count']: raise ValueError('Original batch denominator drift')
    population_rows(queries, 23571)
    if Counter(q['fold'] for q in queries) != {2: 7767, 3: 7967, 4: 7837}:
        raise ValueError('Original full safe-fold denominator drift')
    return queries, geos


def source_worker(root):
    load(root); dest = root/'source'; budget = None
    try:
        with Budget(LIMITS) as budget:
            def guard():
                if budget.failure: raise RuntimeError(budget.failure)
            files = SourceFiles(guard)
            queries, geos = read_sources(files)
            cfg = baseline.config()
            serving = json.loads(files.file(cfg['serving_query_launch']).read_text())
            pilot = list(old_rows(serving['queries']))
            models = json.loads(files.file(serving['models']).read_text()); validate_models(models)
            train = sorted(set().union(*(set(m['query_ids']) for m in models.values())))
            # Reuse only raw geometry, not old candidates/metrics. Assert exact
            # all-field equality on every previously independently resolved row.
            _, base, old, _, _ = baseline.backend.inputs()
            original_geo = json.loads(files.file(old['geo_queries']).read_text())
            if ({g['query_id']: g for g in original_geo} != {q['query_id']: geos[q['query_id']] for q in pilot}
                    or any(q != next(r for r in queries if r['query_id'] == q['query_id']) for q in pilot)):
                raise ValueError('STOP_FULL128_CRM_BAN_PARITY')
            selected = choose(queries, pilot, train)
            meta = dict(full_queries=23571, eligible_queries=22931, excluded_pilot=128, excluded_train=512,
                selected_queries=SIZE, seed=SEED, selection_uses_labels=False,
                old_initial100_panel_overlap=0, training_query_overlap=0,
                folds=dict(Counter(str(q['fold']) for q in selected)),
                departments=dict(Counter(q['crm_insee'][:2] or q['crm_postcode'][:2] or 'UNKNOWN' for q in selected)),
                empty_names=sum(not q['crm_name'] for q in selected), empty_streets=sum(not q['crm_street'] for q in selected),
                source_historical_dev_reused=True, pristine_holdout=False, original128_ban_full_parity=True)
            value = dict(queries=selected, geo_queries=[geos[q['query_id']] for q in selected], metadata=meta,
                models=serving['models'], bundle=serving['bundle'], training_manifest=serving['model_manifest'],
                index_manifest=old['index_manifest'], index_meta=old['index_meta'], inventory=old['inventory'],
                index_documents=old['index_documents'], source_epochs=list(files.entries.values()), **FALSE)
            files.verify_after(); load(root)
            write_json(dest/'panel.json', value)
        validate_report(budget.report())
        write_json(dest/'manifest.json', dict(status='QUARANTINED_V322_SOURCE', panel=record(dest/'panel.json'),
            launch=record(root/'launch.json'), preflight=record(dest/'preflight.json'), resources=budget.report(), **FALSE))
    except BaseException as exc:
        write_json(dest/'worker_stop.json', dict(status='STOP', reason=str(exc), **FALSE)); raise


def stage(root, name):
    load(root); dest = root/name; parent_exit(dest)
    m = json.loads((dest/'manifest.json').read_text()); validate_report(m['resources'])
    gate = json.loads(verify(m['preflight']).read_text())
    if (m['launch'] != record(root/'launch.json') or m['preflight'] != record(dest/'preflight.json')
            or gate['policy_id'] != POLICY or gate['pass_system_quiet_policy'] is not True or gate['quiet_seconds'] < 30
            or gate['worker_swap_budget_changed'] is not True or any(m.get(k) is not False for k in SAFE_FALSE)):
        raise ValueError('Actual own successful full source/encoder producer required')
    key = 'panel' if name == 'source' else 'expansions'
    if m[key] != record(dest/(key+'.json')):
        raise ValueError('Actual own producer artifact path/hash/size required')
    verify(m[key])
    return m


def panel(root):
    root = Path(root); m = stage(root, 'source')
    p = json.loads(verify(m['panel']).read_text()); population_rows(p['queries'])
    if (m['status'] != 'QUARANTINED_V322_SOURCE' or len(p['geo_queries']) != SIZE
            or [(g['query_id'],g['fold']) for g in p['geo_queries']] != [(q['query_id'],q['fold']) for q in p['queries']]
            or p['queries'] != sorted(p['queries'],key=priority) or any(p.get(k) is not False for k in SAFE_FALSE)):
        raise ValueError('All original ordered query-only population required')
    check_epochs(canonical_epochs(p['source_epochs']))
    return p


def encode_worker(root, batch):
    import torch
    from sentence_transformers import SentenceTransformer
    from src.xgb_matcher.semantic import _repair_exported_tokenizer, assert_tokenizer_healthy
    from src.xgb_matcher.retrieval_contrastive_v52 import surface
    from src.xgb_matcher.retrieval_sparse_query_v59 import unit
    dest = root/f'encode{batch:03d}'; budget = None
    try:
        with Budget(LIMITS) as budget:
            p = panel(root); b = json.loads(verify(p['bundle']).read_text())
            models = json.loads(verify(p['models']).read_text()); validate_models(models)
            if b['models'] != p['models'] or b['encoder_prefix'] != 'nom' or b['max_length'] != 128 or b['sparse_terms'] != 8:
                raise ValueError('Same original serving name encoder required')
            for ref in b['model_refs'].values(): verify(ref)
            torch.set_num_threads(1); torch.set_num_interop_threads(1)
            model = SentenceTransformer(b['model_path'],device='cpu',local_files_only=True)
            _repair_exported_tokenizer(model,b['model_path']); assert_tokenizer_healthy(model.tokenizer)
            model.max_seq_length = b['max_length']
            start = batch*128; qs = p['queries'][start:min(start+128,SIZE)]
            if type(batch) is not int or not 0 <= batch < 16 or not qs: raise ValueError('Exact original encoder batch required')
            vectors = unit(model.encode([surface([q['crm_name']],'nom')[0] for q in qs],batch_size=32,
                normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False))
            vocabs = {f: json.loads(verify(m['vocabulary']).read_text()) for f,m in models.items()}
            projections = {f: np.load(verify(m['projection']),allow_pickle=False) for f,m in models.items()}
            if vectors.shape != (len(qs),384): raise ValueError('Original per-query representation required')
            if any(projections[f].shape != (384,len(vocabs[f])) for f in models):
                raise ValueError('Exact original projection/vocabulary dimensions required')
            rows = expansion_rows(vectors,qs,vocabs,projections,8)
            write_json(dest/'expansions.json',dict(rows=rows,panel=record(root/'source/panel.json'), **FALSE))
            for ref in b['model_refs'].values(): verify(ref)
            validate_models(models); panel(root)
        validate_report(budget.report())
        write_json(dest/'manifest.json',dict(status='QUARANTINED_V322_ENCODER',expansions=record(dest/'expansions.json'),
            launch=record(root/'launch.json'),preflight=record(dest/'preflight.json'),resources=budget.report(), **FALSE))
    except BaseException as exc:
        write_json(dest/'worker_stop.json',dict(status='STOP',reason=str(exc), **FALSE)); raise


def expansions(root):
    root = Path(root); p = panel(root); result = []
    for i in range(16):
        m = stage(root,f'encode{i:03d}')
        v = json.loads(verify(m['expansions']).read_text())
        if (m['status'] != 'QUARANTINED_V322_ENCODER' or v['panel'] != record(root/'source/panel.json')
                or any(v.get(k) is not False for k in SAFE_FALSE)):
            raise ValueError('Actual source-bound encoder producer required')
        result.extend(v['rows'])
    if [(r['query_id'],r['fold']) for r in result] != [(q['query_id'],q['fold']) for q in p['queries']]:
        raise ValueError('Every2000 exact ordered expansion before serving required')
    return result


def sealed(root):
    root = Path(root); p = panel(root)
    m = json.loads((root/'manifest.json').read_text())
    if (m['status'] != ENCODED or m['queries'] != SIZE or m['panel'] != record(root/'source/panel.json')
            or m['launch'] != record(root/'launch.json') or m['source'] != record(root/'source/manifest.json')
            or m['encoders'] != [record(root/f'encode{i:03d}/manifest.json') for i in range(16)]
            or any(m.get(k) is not False for k in SAFE_FALSE)):
        raise ValueError('Entire new query package sealed before candidates required')
    v = json.loads(verify(m['expansions']).read_text())
    if v['rows'] != expansions(root) or v['panel'] != m['panel'] or any(v.get(k) is not False for k in SAFE_FALSE):
        raise ValueError('Every actual expansion bound to same panel required')
    return p,m,v


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT): raise ValueError('Fresh dedicated panel root required')
    root.mkdir(parents=True)
    write_json(root/'launch.json',dict(seed=SEED,queries=SIZE,implementation=implementation(), **FALSE))
    try:
        for name in ['source', *[f'encode{i:03d}' for i in range(16)]]:
            dest = root/name; dest.mkdir(); write_json(dest/'preflight.json',quiet(30,50))
            cmd = [sys.executable,'-u','-m','scripts.retrieval_v322_large_panel', 'source' if name=='source' else 'encode',
                '--output',str(root)]
            if name!='source': cmd += ['--batch',str(int(name[6:]))]
            with (dest/'worker.log').open('x') as log:
                s = supervise(cmd,log,ENV,dest/'supervisor_stop.json')
            write_json(dest/'supervision.json',s); stage(root,name)
            if name=='source':
                write_json(root/'population_manifest.json',dict(status=STATUS,panel=record(root/'source/panel.json'),
                    source=record(root/'source/manifest.json'),launch=record(root/'launch.json'), **FALSE))
                print(json.dumps(panel(root)['metadata']),flush=True)
            else: print(json.dumps(dict(encoded=min((int(name[6:])+1)*128,SIZE))),flush=True)
        value = expansions(root)
        write_json(root/'expansions.json',dict(rows=value,panel=record(root/'source/panel.json'), **FALSE))
        write_json(root/'manifest.json',dict(status=ENCODED,queries=SIZE,panel=record(root/'source/panel.json'),
            source=record(root/'source/manifest.json'),encoders=[record(root/f'encode{i:03d}/manifest.json') for i in range(16)],
            expansions=record(root/'expansions.json'),launch=record(root/'launch.json'), **FALSE))
        sealed(root)
    except BaseException as exc:
        write_json(root/'status.json',dict(status='STOP',reason=str(exc), **FALSE)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','source','encode'));parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batch',type=int);a=parser.parse_args()
    if a.mode=='run':run(a.output)
    elif a.mode=='source':source_worker(a.output)
    else:encode_worker(a.output,a.batch)
