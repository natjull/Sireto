"""Strict loader for bounded sparse TRAIN probes, never a retrieval evaluator."""
import json

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts.preflight_retrieval_v69_sparse_contrastive import validated_alias_source
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend, normalize_text
from src.xgb_matcher.retrieval_joint_contrastive_v63 import unit_parts
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_v20_workbench import verify


def load(config, launch):
    source = json.loads(verify(config['vector_source']).read_text())
    sl = json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
    data = json.loads(verify(sl['inputs']).read_text()); cache = json.loads(verify(source['worker']).read_text())
    proof = json.loads(verify(config['proof_source']).read_text()); v38.verify_launch(proof)
    validated_alias_source(config['alias_validation'])
    rows = data['rows']
    scenes = [s for ref in proof['features'] for s in json.loads(verify(ref).read_text())['scenes']]
    if source['recall_measured'] or proof['recall_measured'] or len(rows) != 512 or len(scenes) != 512:
        raise ValueError('Training-only source denominator drift')
    if cache['query_ids'] != [r['query_id'] for r in rows] or any(
        any(r[k] != s[k] for k in ('query_id', 'fold', 'positive_siren')) or
        len(r['negatives']) != len(s['negatives']) for r, s in zip(rows, scenes)):
        raise ValueError('Scene/query source order drift')
    if any(r['fold'] not in (2, 3, 4) for r in rows) or set(data['registry']) != {'2', '3', '4'}:
        raise ValueError('Closed fold in training source')
    q = unit_parts(np.load(verify(cache['query_vectors']), allow_pickle=False))
    for i, r in enumerate(rows):
        for view, field in enumerate(('name', 'address')):
            if not data['query_fields'][r['query_id']][field].split(':', 1)[-1].strip():
                q[i, view] = 0.
    backend = TantivyBackend(v38.INDEX); tv = backend._tantivy
    if backend.searcher.num_docs != launch['index_documents']:
        raise ValueError('National official index count drift')
    ids = cache['document_ids']; documents = {}; aliases = {}
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate official training doc')
    for start in range(0, len(ids), 2048):
        subset = ids[start:start+2048]
        for _, loc in backend.searcher.search(tv.Query.term_set_query(backend.index.schema, 'siret', subset),
                                             limit=len(subset)+1, count=False).hits:
            d = backend.searcher.doc(loc).to_dict(); sid = backend._first(d, 'siret')
            if sid not in subset or sid in documents or not sid.startswith(backend._first(d, 'siren')):
                raise ValueError('Official source identity drift')
            documents[sid] = {f:d.get(f, []) for f in (*v38.NAMES, *v38.ADDRESSES)}
            aliases[sid] = [v for f in ('legal_names', 'trade_names', 'site_names') for v in d.get(f, [])]
    if set(documents) != set(ids):
        raise ValueError('Missing official doc; no denominator filtering')
    exact_audited = 0
    for r, s in zip(rows, scenes):
        name = signal(r['crm_name'])
        for sid, f in [(r['positive_siret'], s['positive_features']),
            *[(n['siret'], nf['features']) for n, nf in zip(r['negatives'], s['negatives'])]]:
            if float(bool(name and name in {normalize_text(a) for a in aliases[sid]})) != f[2]:
                raise ValueError('Stored current alias/native exact proof drift')
            exact_audited += 1
    return q, rows, scenes, documents, aliases, data['registry'], dict(
        official_training_documents=len(ids), current_exact_vectors_audited=exact_audited)
