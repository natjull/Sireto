"""Label-free, bounded Mac feasibility probe. NOT a recall evaluation.

Pinned public model; read-only CRM/official texts from the active Gemma scope.
Never reads candidate outputs, GT, folds 0/1 or the held-out test. Does not
modify the active runtime, model, code, configuration or benchmark files.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/Volumes/CATNAT_DATA/SIRETO_PPLX')
OUTPUT = ROOT / 'probe_20261008_b'
SOURCE = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v14/representative_20261008_b')
REVISION = '8fc2de24534aa3610d85fa59c463313a5f096455'
MODEL = ROOT / 'hf_cache/models--perplexity-ai--pplx-embed-v2-late-0.6b/snapshots' / REVISION


def safe(value):
    return re.sub(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)', ' ', str(value)).strip()


def query_text(row):
    if row['fold'] not in (2, 3, 4):
        raise ValueError('Closed fold')
    return safe('noms: ' + row['crm_name'] + ' adresse: ' + ' '.join(
        row[k] for k in ('crm_number', 'crm_suffix', 'crm_street', 'crm_postcode')))


def official_text(doc):
    extra = doc['extra']
    atoms = (['nom entreprise: ' + n for n in extra['legal']]
        + ['nom etablissement: ' + n for n in doc['own_names']]
        + ['ancien nom etablissement: ' + n for n in extra['own_site']]
        + ['adresse etablissement: ' + a for a in doc['all_addresses']]
        + ['code postal: ' + doc['postcode']])
    return ' | '.join(dict.fromkeys(safe(a) for a in atoms))


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    path = OUTPUT / name
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inputs():
    panel = read(SOURCE / 'panel.json')
    zones = read(SOURCE / 'census.json')['zones']
    # Small prepared zones only for bounded hardware measurement, NOT representativity.
    available = [z for z in zones if 0 < z['indexed_sites'] <= 10000
        and (SOURCE / 'zones' / z['zone'] / 'documents.seal.json').exists()
        and any(q['crm_insee'] == z['zone'] for q in panel)]
    available.sort(key=lambda z: hashlib.sha256(('pplx-probe-v1|' + z['zone']).encode()).digest())
    queries, documents, refs = [], [], []
    for zone in available[:4]:
        receipt = read(SOURCE / 'zones' / zone['zone'] / 'prepare_exit.json')
        if receipt.get('exit_code') != 0 or not receipt.get('own_child_exit_confirmed'):
            raise ValueError('Unowned/incomplete official preparation')
        ref = read(SOURCE / 'zones' / zone['zone'] / 'documents.seal.json')
        path = Path(ref['path'])
        if sha(path) != ref['sha256']:
            raise ValueError('Official document seal drift')
        docs = read(path)
        docs.sort(key=lambda d: hashlib.sha256(('pplx-doc-v1|' + str(d['key'])).encode()).digest())
        documents.extend(official_text(d) for d in docs[:32])
        local = [q for q in panel if q['crm_insee'] == zone['zone']]
        local.sort(key=lambda q: hashlib.sha256(str(q['query_id']).encode()).digest())
        queries.extend(query_text(q) for q in local[:4])
        refs.append(dict(zone=zone['zone'], full_sites=len(docs), source=ref))
    if not queries or not documents:
        raise ValueError('No prepared query-only inputs')
    return queries, documents, refs


def run():
    import importlib.metadata
    import psutil
    import torch
    from sentence_transformers import MultiVectorEncoder
    from scripts.embeddinggemma2_night import pressure
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    save('state.json', dict(status='PREPARING_LABEL_FREE_TECHNICAL_PROBE', pid=os.getpid()))
    model = None
    started = time.monotonic()

    def guard(admit=False):
        free, level = pressure()
        gpu = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
        if free < (30 if admit else 20) or level not in (0, 1, 2) or gpu > 6 * 1024**3:
            raise RuntimeError(f'Resource guard free={free} pressure={level} mps={gpu}')
        if time.monotonic() - started > 180:
            raise RuntimeError('Bounded technical probe timeout, not a retrieval result')

    try:
        guard(True)
        if not torch.backends.mps.is_available():
            raise RuntimeError('Metal/MPS unavailable')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
            raise RuntimeError('CPU fallback must be disabled for honest Metal test')
        queries, docs, refs = inputs()
        save('inputs.json', dict(queries=queries, documents=docs, official_sources=refs,
            truth_opened=False, small_communes_hardware_only=True))
        save('state.json', dict(status='LOADING_PINNED_MODEL', pid=os.getpid()))
        begin = time.monotonic()
        model = MultiVectorEncoder(str(MODEL), local_files_only=True, device='mps',
            trust_remote_code=False, model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'})
        load_seconds = time.monotonic() - begin
        # No undocumented Gemma prompts, quantization, token pooling or silent truncation.
        tokenizer = model.tokenizer
        lengths = [len(tokenizer(t, add_special_tokens=True)['input_ids']) for t in queries + docs]
        if max(lengths) + 32 > model.max_seq_length:
            raise ValueError('Input would truncate: stop instead of reporting a misleading speed')
        guard()
        torch.mps.synchronize()
        begin = time.monotonic()
        warm = model.encode_document(docs[:1], batch_size=1, show_progress_bar=False)
        torch.mps.synchronize()
        warm_seconds = time.monotonic() - begin
        del warm
        times, vectors, peak_driver, peak_current = [], [], 0, 0
        for lo in range(0, len(docs), 4):
            guard()
            begin = time.monotonic()
            encoded = model.encode_document(docs[lo:lo+4], batch_size=4, show_progress_bar=False)
            torch.mps.synchronize()
            times.append(time.monotonic() - begin)
            vectors.extend(v.detach().float().cpu() for v in encoded)
            peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
            peak_current = max(peak_current, torch.mps.current_allocated_memory())
            del encoded
            save('state.json', dict(status='ENCODING_TECHNICAL_SAMPLE', pid=os.getpid(),
                completed=len(vectors), total=len(docs)))
        guard()
        begin = time.monotonic()
        qv = model.encode_query(queries, batch_size=4, show_progress_bar=False)
        torch.mps.synchronize()
        query_seconds = time.monotonic() - begin
        qv = [v.detach().float().cpu() for v in qv]
        if any(v.ndim != 2 or v.shape[1] != 128 or not torch.isfinite(v).all() for v in qv + vectors):
            raise ValueError('Invalid multi-vector embedding')
        begin = time.monotonic()
        scores = model.similarity(qv, vectors)
        score_seconds = time.monotonic() - begin
        if tuple(scores.shape) != (len(queries), len(docs)) or not torch.isfinite(scores).all():
            raise ValueError('Invalid MaxSim output')
        # Independent scalar oracle checks native scoring on two real pairs.
        oracle_errors = []
        for j in range(min(2, len(vectors))):
            expected = (qv[0] @ vectors[j].T).max(dim=1).values.sum()
            oracle_errors.append(abs(float(expected) - float(scores[0, j])))
        if max(oracle_errors) > 1e-3:
            raise ValueError(f'MaxSim oracle disagreement {oracle_errors}')
        footprint = subprocess.run(['/usr/bin/vmmap', '-summary', str(os.getpid())],
            capture_output=True, text=True, timeout=10)
        save('memory.json', dict(returncode=footprint.returncode, vmmap_summary=footprint.stdout))
        report = dict(status='COMPLETE_TECHNICAL_ONLY_NO_RECALL', model_revision=REVISION,
            implementation_sha256=sha(Path(__file__)), device='mps', cpu_fallback=False,
            dtype='float32', attention='eager', batch_size=4, queries=len(queries), documents=len(docs),
            load_seconds=load_seconds, warmup_seconds=warm_seconds,
            document_seconds=sum(times), documents_per_second=len(docs)/sum(times),
            query_seconds=query_seconds, cpu_maxsim_seconds=score_seconds,
            maxsim_pairs=len(docs)*len(queries), oracle_max_error=max(oracle_errors),
            token_lengths=lengths, document_vector_counts=[len(v) for v in vectors],
            document_vectors_bytes=sum(v.numel()*v.element_size() for v in vectors),
            mps_peak_observed_driver_bytes=peak_driver, mps_peak_observed_current_bytes=peak_current,
            rss_bytes=psutil.Process().memory_info().rss,
            versions={p: importlib.metadata.version(p) for p in ('torch','transformers','sentence-transformers','pillow','torchvision')},
            labels_read=False, retrieval_evaluated=False, serving_or_gemma_changed=False)
        save('report.json', report)
        save('state.json', dict(status=report['status'], pid=os.getpid()))
        print(json.dumps(report), flush=True)
    except Exception as error:
        save('state.json', dict(status='STOP_TECHNICAL_PROBE', error_type=type(error).__name__,
            error=str(error), labels_read=False, pid=os.getpid()))
        raise
    finally:
        del model


if __name__ == '__main__':
    run()
