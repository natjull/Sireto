"""Label-blind official encoding cost preflight; never computes recall."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.xgb_matcher.retrieval_learned_v41 import official_text, unit_vectors
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json


def prepare(config):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = verify(config['source_manifest'])
    manifest = json.loads(source.read_text())
    if manifest['policy']['crm_labels_present'] or manifest['policy']['model_scores_present']:
        raise ValueError('Non-official source')
    root = source.parent
    refs = {name: dict(path=str(root/name), **manifest['files'][name])
            for name in ('name_evidence.parquet', 'establishments.parquet')}
    for ref in refs.values():
        verify(ref)
    names = pq.ParquetFile(root/'name_evidence.parquet')
    groups = sorted(range(names.num_row_groups), key=lambda i: hashlib.sha256(f'v41-cost|{i}'.encode()).digest())
    rows, used = {}, []
    columns = ['siren', 'siret', 'subject_kind', 'normalized_value', 'evidence_id', 'source']
    for group in groups[:32]:
        used.append(group)
        batch = next(names.iter_batches(batch_size=4096, row_groups=[group], columns=columns))
        for r in batch.to_pylist():
            if r['subject_kind'] == 'SIRET' and r['source'] == 'SIRENE_CURRENT' and r['normalized_value']:
                rows.setdefault(r['siret'], r)
            if len(rows) == config['sample_rows']:
                break
        if len(rows) == config['sample_rows']:
            break
    if len(rows) != config['sample_rows']:
        raise ValueError('Insufficient label-blind official name sample')
    conn = duckdb.connect()
    conn.execute("SET threads=2")
    conn.execute("SET memory_limit='512MB'")
    conn.execute("SET max_temp_directory_size='0B'")
    conn.register('sample', pa.Table.from_pylist(list(rows.values())))
    sites_path = str(root/'establishments.parquet').replace("'", "''")
    try:
        bound = conn.execute(f"""SELECT n.siret, n.evidence_id, n.normalized_value,
            s.current_address_normalized FROM sample n
            JOIN read_parquet('{sites_path}') s ON n.siret=s.siret AND n.siren=s.siren
            ORDER BY n.siret""").fetchall()
    finally:
        conn.close()
    if len(bound) != config['sample_rows']:
        raise ValueError('Official membership/sample cardinality failure')
    texts = [official_text([r[2]], r[3] or '') for r in bound]
    return dict(texts=texts, provenance=[dict(siret=r[0], evidence_id=r[1]) for r in bound],
                row_groups=used, source_refs=refs, national_sites=manifest['counts']['establishments'])


def encode_worker(output):
    config = json.loads((output/'launch.json').read_text())['config']
    inputs = json.loads((output/'inputs.json').read_text())
    # Torch is isolated from DuckDB/PyArrow/FAISS OpenMP runtimes.
    import torch
    from sentence_transformers import SentenceTransformer
    from src.xgb_matcher.semantic import _repair_exported_tokenizer, assert_tokenizer_healthy
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    with Budget(config['resources']) as budget:
        started = time.monotonic()
        model = SentenceTransformer(config['model_path'], device=config['device'], local_files_only=True)
        _repair_exported_tokenizer(model, config['model_path'])
        assert_tokenizer_healthy(model.tokenizer)
        model.max_seq_length = 128
        load_seconds = time.monotonic()-started
        texts = inputs['texts']
        model.encode(texts[:16], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        started = time.monotonic()
        vectors = unit_vectors(model.encode(texts, batch_size=32, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False))
        encode_seconds = time.monotonic()-started
        single_seconds = []
        for text in texts[:16]:
            start = time.monotonic()
            model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            single_seconds.append(time.monotonic()-start)
        token_counts = [len(model.tokenizer(t)['input_ids']) for t in texts]
        np.save(output/'vectors.npy', vectors, allow_pickle=False)
    sites, dim = inputs['national_sites'], vectors.shape[1]
    write_json(output/'worker.json', dict(
        rows=len(texts), dimension=dim, load_seconds=load_seconds, encode_seconds=encode_seconds,
        rows_per_second=len(texts)/encode_seconds, query_encode_p50=float(np.quantile(single_seconds, .5)),
        query_encode_p95=float(np.quantile(single_seconds, .95)),
        token_counts_p50=float(np.quantile(token_counts, .5)), token_counts_p95=float(np.quantile(token_counts, .95)),
        max_length=128, truncated_rows=sum(n>128 for n in token_counts),
        national_single_view_encoding_hours=sites*encode_seconds/len(texts)/3600,
        national_float32_vector_bytes=sites*dim*4, national_48byte_codes_bytes=sites*48,
        ann_code_bytes_are_storage_arithmetic_not_measured_quality=True,
        resources=budget.report(), vectors=record(output/'vectors.npy'),
        truth_opened=False, recall_measured=False))


def run(output, config_path):
    config = json.loads(config_path.read_text())
    if config['schema'] != 'sireto-v41-learned-initial-search-preflight-1' or config['sample_rows'] != 512:
        raise ValueError('Scope drift')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with Budget(config['resources']) as prep_budget:
            inputs = prepare(config)
        write_json(output/'inputs.json', inputs)
        model_root = Path(config['model_path'])
        model_refs = {str(p.relative_to(model_root)): record(p) for p in sorted(model_root.rglob('*')) if p.is_file()}
        if not model_refs:
            raise ValueError('Missing local model')
        launch = dict(config=config, config_ref=record(config_path), model_refs=model_refs,
                      inputs=record(output/'inputs.json'), implementation={str(p):record(p) for p in
                        (Path(__file__), ROOT/'src/xgb_matcher/retrieval_learned_v41.py',
                         ROOT/'src/xgb_matcher/semantic.py', ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')})
        write_json(output/'launch.json', launch)
        env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false',
               'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2', 'VECLIB_MAXIMUM_THREADS':'2'}
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable, '-u', '-m', 'scripts.preflight_retrieval_learned_v41',
                            'worker', '--output', str(output)], env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=max(1,config['resources']['max_seconds']-(time.monotonic()-started)))
        for ref in [launch['config_ref'], launch['inputs'], *launch['model_refs'].values(),
                    *launch['implementation'].values()]:
            verify(ref)
        for ref in inputs['source_refs'].values():
            # Already fully hashed before preparation. No source writes occur.
            if Path(ref['path']).stat().st_size != ref['size_bytes']:
                raise ValueError('Official source changed')
        write_json(output/'manifest.json', dict(status='SEALED_ENCODING_COST_ONLY', launch=record(output/'launch.json'),
            worker=record(output/'worker.json'), preparation_resources=prep_budget.report(),
            seconds=time.monotonic()-started, truth_opened=False, recall_measured=False,
            ready_for_fold1=False, national_build_launched=False))
        print((output/'worker.json').read_text(), flush=True)
    except BaseException as exc:
        write_json(output/'status.json', dict(status='STOP', reason=str(exc), truth_opened=False, recall_measured=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'worker'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/retrieval_v41_learned_preflight.json')
    args = parser.parse_args()
    run(args.output, args.config) if args.mode == 'run' else encode_worker(args.output)
