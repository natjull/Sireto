"""Query-only national hard negatives on a separate TRAIN64, never product recall."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

from scripts import retrieval_v217_corroborated_population as base
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend, RetrievalQuery
from src.xgb_matcher.retrieval_proof_v56 import components, index_query, point_features
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles, identity
from src.xgb_matcher.retrieval_memory_policy_v214 import Budget, LIMITS, quiet, validate_report, parent_exit
from src.xgb_matcher.retrieval_supervisor_v214 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

OUTPUT = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v227_national_training_negatives')
SOURCE = ROOT/'config/retrieval_v77_safe_sparse.json'
FIELDS = ('query_id', 'fold', 'crm_name', 'crm_number', 'crm_suffix', 'crm_street', 'crm_postcode', 'crm_insee')
STATUS = 'SEALED_V227_TRAIN64_QUERY_ONLY_NATIONAL_NEGATIVES_NOT_RECALL'


def project(rows, controls):
    """Pre-existing source order; selection never consults a positive or negative."""
    if len(rows) != 512 or len({r['query_id'] for r in rows}) != 512:
        raise ValueError('Entire original512 training source required')
    result = [{k: r[k] for k in FIELDS} for r in rows[:64]]
    if ({r['query_id'] for r in rows} & {r['query_id'] for r in controls}
            or any(type(r['fold']) is not int or r['fold'] not in (2, 3, 4) for r in result)
            or any(not isinstance(r[k], str) for r in result for k in FIELDS if k != 'fold')):
        raise ValueError('Safe folds, raw CRM strings and disjoint TRAIN queries required')
    return result


def as_query(raw):
    return RetrievalQuery(name=raw['crm_name'], address=' '.join(raw[k] for k in ('crm_number', 'crm_suffix', 'crm_street')),
                          number=raw['crm_number'], postcode=raw['crm_postcode'], insee=raw['crm_insee'])


def implementation():
    paths = [Path(__file__), ROOT/'docs/retrieval_v227_national_training_contract.md', SOURCE,
             ROOT/'src/xgb_matcher/retrieval_proof_v56.py',
             ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
             ROOT/'src/xgb_matcher/retrieval_memory_policy_v214.py',
             ROOT/'src/xgb_matcher/retrieval_supervisor_v214.py']
    return {str(p): record(p) for p in paths}


def load(root):
    if (not root.resolve().is_relative_to(OUTPUT) or (root/'status.json').exists()
            or any(root.glob('batch*/supervisor_stop.json'))):
        raise ValueError('Original fresh non-STOP mining root required')
    l = json.loads((root/'launch.json').read_text())
    for ref in l['implementation'].values(): verify(ref)
    if (l['queries'] != 64 or l['resources'] != LIMITS or l['truth_opened'] is not False
            or l['recall_measured'] is not False or l['downstream_models_changed'] is not False):
        raise ValueError('Query-only64 training miner, unchanged protections required')
    rows = json.loads(verify(l['raw_queries']).read_text())
    if len(rows) != 64 or len({r['query_id'] for r in rows}) != 64 or any(set(r) != set(FIELDS) for r in rows):
        raise ValueError('Strict raw query projection without targets required')
    for ref in (l['index_manifest'], l['index_meta']): verify(ref)
    if base.backend.v38.inventory() != l['inventory']:
        raise ValueError('Native inventory changed')
    return l, rows


def valid(rows, queries):
    if len(rows) != len(queries): raise ValueError('No partial training batch admission')
    for row, q in zip(rows, queries):
        if (row['query_id'] != q['query_id'] or row['fold'] != q['fold'] or not 0 <= row['seconds'] <= 20
                or set(row['channels']) != {'name', 'joint'}):
            raise ValueError('Actual aligned full training query and time required')
        for values in row['channels'].values():
            if len(values) > 100 or len({c['siret'] for c in values}) != len(values):
                raise ValueError('Each independent national mining collector <=100')
            for c in values: identity(c)
        ids = {c['siret'] for values in row['channels'].values() for c in values}
        if set(row['features']) != ids or any(len(v) != 11 for v in row['features'].values()):
            raise ValueError('Point features only on naturally mined references')


def worker(root, batch):
    l, queries = load(root); dest = root/f'batch{batch:03d}'
    if batch not in range(8): raise ValueError('Eight physical batches only')
    rows = []
    with Budget(LIMITS) as budget:
        def guard():
            if budget.failure: raise RuntimeError(budget.failure)
        files = SourceFiles(guard)
        models = json.loads(files.file(l['models']).read_text()); base.validate_models(models)
        for raw in queries[batch*8:batch*8+8]:
            t = time.monotonic(); native = TantivyBackend(base.backend.v38.INDEX)
            if native.searcher.num_docs != l['index_documents']: raise ValueError('Native cardinality changed')
            q = as_query(raw); w = models[str(raw['fold'])]['weights']; tv = native._tantivy
            c = components(native, q)
            name = [tv.Query.boost_query(c[i], float(w[i])) for i in (0, 2) if c[i] is not None and w[i] > 0]
            leaf = tv.Query.const_score_query(tv.Query.term_query(native.index.schema, 'document_type', 'siret', 'basic'), 0.)
            parsed = tv.Query.boolean_query([(tv.Occur.Must, leaf), (tv.Occur.Must,
                tv.Query.boolean_query([(tv.Occur.Should, p) for p in name]))]) if name else tv.Query.empty_query()
            channels = {}
            for key, query in [('name', parsed), ('joint', index_query(native, q, w))]:
                found = []
                for score, loc in native.searcher.search(query, limit=100, count=False).hits:
                    d = native.searcher.doc(loc).to_dict()
                    found.append(dict(siret=native._first(d, 'siret'), siren=native._first(d, 'siren')))
                channels[key] = found; guard()
            ids, vectors = point_features(native, q, [c['siret'] for values in channels.values() for c in values])
            elapsed = time.monotonic()-t
            if elapsed > 20: raise TimeoutError('STOP_MINING_QUERY_20_SECONDS')
            rows.append(dict(query_id=raw['query_id'], fold=raw['fold'], seconds=elapsed,
                             channels=channels, features={s: v.tolist() for s, v in zip(ids, vectors)}))
            print(json.dumps(dict(query_id=raw['query_id'], seconds=elapsed, references=len(ids))), flush=True)
            native = None; gc.collect(); guard()
        files.verify_after(); load(root); valid(rows, queries[batch*8:batch*8+8])
        write_json(dest/'payload.json', dict(rows=rows, source_epochs=list(files.entries.values()), truth_opened=False))
    validate_report(budget.report())
    write_json(dest/'manifest.json', dict(resources=budget.report(), payload=record(dest/'payload.json'),
        launch=record(root/'launch.json'), preflight=record(dest/'preflight.json')))


def admission(root):
    l, queries = load(root); rows = []; producers = []; epochs = []
    for batch in range(8):
        dest = root/f'batch{batch:03d}'; parent_exit(dest)
        m = json.loads((dest/'manifest.json').read_text()); validate_report(m['resources'])
        gate = json.loads(verify(m['preflight']).read_text())
        if (m['launch'] != record(root/'launch.json') or m['preflight'] != record(dest/'preflight.json')
                or gate['pass_system_quiet_policy'] is not True or gate['quiet_seconds'] < 30
                or m['payload'] != record(dest/'payload.json')):
            raise ValueError('Actual producer launch, preflight and output required')
        p = json.loads(verify(m['payload']).read_text()); valid(p['rows'], queries[batch*8:batch*8+8])
        if p['truth_opened'] is not False: raise ValueError('Mining labels must stay closed')
        rows.extend(p['rows']); epochs.extend(p['source_epochs']); producers.append(record(dest/'manifest.json'))
    base.check_epochs(epochs); valid(rows, queries)
    return dict(rows=rows, producers=producers, source_epochs=epochs)


def sealed(root):
    value = admission(root); m = json.loads((root/'manifest.json').read_text())
    if m['status'] != STATUS or m['launch'] != record(root/'launch.json') or m['producers'] != value['producers']:
        raise ValueError('All64 mining sealed before any trainer point truth')
    if json.loads(verify(m['outputs']).read_text()) != value or m['outputs'] != record(root/'outputs.json'):
        raise ValueError('Same complete mining output required')
    return value


def run(root):
    if root.exists() or not root.resolve().is_relative_to(OUTPUT): raise ValueError('Fresh dedicated root required')
    _, _, _, controls, _ = base.backend.inputs()
    cfg = json.loads(verify(record(SOURCE)).read_text())
    proof = json.loads(verify(cfg['proof_source']).read_text()); old = json.loads(verify(proof['launch']).read_text())
    if proof['training_queries'] != 512 or old['truth_scope'] != 'TRAINER_ONLY': raise ValueError('Original TRAIN source required')
    queries = project(old['training_rows'], controls)
    serving = json.loads(verify(base.config()['serving_query_launch']).read_text())
    root.mkdir(parents=True)
    write_json(root/'raw_queries.json', queries)
    write_json(root/'training_authority.json', dict(source_config=record(SOURCE), source_proof=cfg['proof_source'],
        source_launch=proof['launch'], raw_queries=record(root/'raw_queries.json'),
        selection='FIRST64_OF_ORIGINAL512_ORDER_LABEL_BLIND', supervised_truth_scope='TRAINER_ONLY'))
    write_json(root/'launch.json', dict(queries=64, resources=LIMITS, implementation=implementation(),
        raw_queries=record(root/'raw_queries.json'), models=serving['models'],
        index_manifest=old['index_manifest'], index_meta=old['index_meta'], inventory=old['inventory'],
        index_documents=old['index_documents'], truth_opened=False, recall_measured=False, downstream_models_changed=False))
    try:
        for batch in range(8):
            dest = root/f'batch{batch:03d}'; dest.mkdir()
            write_json(dest/'preflight.json', quiet(30, 50))
            with (dest/'worker.log').open('x') as log:
                s = supervise([sys.executable, '-u', '-m', 'scripts.retrieval_v227_national_training_negatives',
                    'worker', '--output', str(root), '--batch', str(batch)], log,
                    {**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'RAYON_NUM_THREADS':'1',
                     'VECLIB_MAXIMUM_THREADS':'1'}, dest/'supervisor_stop.json')
            write_json(dest/'supervision.json', s); parent_exit(dest)
            print(json.dumps(dict(completed=(batch+1)*8, mode='TRAIN_MINING_NOT_RECALL')), flush=True)
        value = admission(root); write_json(root/'outputs.json', value)
        write_json(root/'manifest.json', dict(status=STATUS, launch=record(root/'launch.json'),
            training_authority=record(root/'training_authority.json'), outputs=record(root/'outputs.json'),
            producers=value['producers'], queries=64, recall_measured=False, ready_for_fold1=False))
        sealed(root)
    except BaseException as exc:
        write_json(root/'status.json', dict(status='STOP', reason=str(exc), recall_measured=False)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('run','worker'))
    p.add_argument('--output', type=Path, required=True); p.add_argument('--batch', type=int)
    a = p.parse_args(); run(a.output) if a.mode == 'run' else worker(a.output, a.batch)
