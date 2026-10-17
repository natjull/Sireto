"""Brief9: bounded, query-only reconstruction, STOP at the first failed fidelity."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ijson
import numpy as np
import psutil
from scripts import train3000_frozen_v142_bano as bano
from scripts.retrieval_site_operational_metric import read, record

OUT = bano.cohort.OUT/'brief9'
LIMIT = 4*1024**3
# Keep admission headroom rather than intentionally approach the user's strict cap.
WATCHDOG = int(3.5*1024**3)
STOP = False


def save(path, value):
    bano.cohort.save(path, value)


def guard():
    rss = psutil.Process().memory_info().rss
    if rss >= WATCHDOG:
        raise MemoryError('WAIT_RESOURCES_BEFORE_4_GIB')
    return rss


def verified(ref):
    result = record(ref['path'], ref['sha256'])
    guard()
    return Path(result['path'])


def subset(ref, ids):
    found = {}
    with verified(ref).open('rb') as stream:
        for qid, value in ijson.kvitems(stream, '', use_float=True):
            if qid in ids:
                found[qid] = value
            guard()
    if set(found) != set(ids):
        raise ValueError('STOP_INCOMPLETE_FROZEN_QUERY_CONTEXT')
    return found


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def bano_fidelity(directory):
    start = time.monotonic()
    run = bano.cohort.preflight.RUN
    launch = read(run/'launch.json')
    sample = read(bano.cohort.OUT/'sample.json')
    panel = read(bano.cohort.OUT/'pilot_fidelity_queries.json')
    ids = set(sample['fidelity_query_ids'])
    if len(panel) != 20 or {q['query_id'] for q in panel} != ids:
        raise ValueError('Sealed twenty-query scope required')
    original_panel = read(verified(launch['panel']))
    if {q['query_id']:q for q in original_panel if q['query_id'] in ids} != {q['query_id']:q for q in panel}:
        raise ValueError('STOP_PILOT_QUERY_BINDING')
    bg = read(verified(launch['bano_generation']))
    bl = read(verified(bg['launch']))
    for ref in (bl['commune_reference'], bl['delegated_reference'], bl['shared_preparation_module']):
        verified(ref)
    for path, sha in bano.HELPERS.items():
        verified(dict(path=str(ROOT/path), sha256=sha))
    build, source, base, kernel_sha = bano.context_builder()
    # Counter is an original constructor global, absent from the earlier adapter's
    # restricted compile namespace. Supply it, without changing the source body.
    build.__globals__['Counter'] = Counter
    with sqlite3.connect(f"file:{verified(bg['index'])}?mode=ro&immutable=1", uri=True) as con:
        con.execute('PRAGMA query_only=ON')
        fresh = build(panel, con)
    frozen = subset(bg['query_context'], ids)
    same = {qid:canonical(fresh[qid]) == canonical(frozen[qid]) for qid in ids}
    save(directory/'query_context.json', fresh)
    if not all(same.values()):
        save(directory/'failure.json', dict(status='STOP_BANO_CONTEXT_NONFIDELITY',
            context_matches=sum(same.values()), failed_ids=sorted(q for q in ids if not same[q])))
        raise ValueError('STOP_BANO_CONTEXT_NONFIDELITY')
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    from scripts.retrieval_miss_score_explain import preparer_kernel
    kernel, delta_sha = preparer_kernel(v142, 'bano')
    fg, prepared, cal = (read(run/p) for p in ('features_generation.json', 'prepared.json', 'calibration/generation.json'))
    assignment = read(verified(cal['assignment']))
    keys_all = np.load(verified(launch['keys']), mmap_mode='r', allow_pickle=False)
    xy = np.load(verified(launch['xy']), mmap_mode='r', allow_pickle=False)
    public = np.load(verified(launch['public_mask']), mmap_mode='r', allow_pickle=False)
    rows = []
    for q in panel:
        guard()
        qid = q['query_id']
        model = read(verified(cal['models'][str(assignment[qid])]))
        if qid in model['train_queries'] or qid not in model['heldout_queries']:
            raise ValueError('STOP_MODEL_NOT_HELDOUT')
        with np.load(verified(fg['features'][qid]), allow_pickle=False) as f:
            keys, features = f['keys'], f['features']
        points, precision = v142.compatible_points(fresh[qid])
        delta = kernel(keys, features, keys_all, xy, public, points, model)
        with np.load(verified(prepared['shards'][qid]), allow_pickle=False) as f:
            same_keys = np.array_equal(keys, f['keys'])
            bitwise = same_keys and np.array_equal(delta, f['bano_geometry'])
            error = float(np.max(np.abs(delta-f['bano_geometry']), initial=0.)) if same_keys else None
        row = dict(query_id=qid, context_identical=same[qid], keys_identical=same_keys,
            delta_identical=bitwise, max_absolute_error=error, precision=precision,
            rows=len(keys), elapsed_seconds=time.monotonic()-start)
        save(directory/(hashlib.sha256(qid.encode()).hexdigest()+'.json'), row)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not bitwise:
            save(directory/'failure.json', dict(status='STOP_BANO_DELTA_NONFIDELITY', query=row))
            raise ValueError('STOP_BANO_DELTA_NONFIDELITY')
    result = dict(status='PASS', family='bano', queries=20, context_matches=sum(same.values()),
        weighted_delta_matches=20, max_absolute_error=max(r['max_absolute_error'] for r in rows),
        context_body_sha256=kernel_sha, weighted_body_sha256=delta_sha, source=source,
        upstream_numeric51='FROZEN_INPUT_FOR_ISOLATED_BANO_TEST_NOT_REBUILT_HERE',
        elapsed_seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        raw_labels_read=False, native_queries=0)
    save(directory/'result.json', result)
    return result


def interrupted(*args):
    global STOP
    STOP = True


def run(family):
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    directory = OUT/(family+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', family, '--directory', str(directory)]
    started = time.monotonic()
    peak = 0
    reason = None
    with (directory/'worker.log').open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        save(directory/'controller.json', dict(pid=os.getpid(), child_pid=child.pid, command=command,
            implementation=record(Path(__file__)), rss_cap_bytes=LIMIT, admission_stop_bytes=WATCHDOG,
            hydraulic_action='NONE; brief9 explicitly permits this mini under4GiB'))
        try:
            while child.poll() is None:
                try:
                    process = psutil.Process(child.pid)
                    rss = sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running())
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                if peak >= WATCHDOG or STOP:
                    reason = 'WAIT_RESOURCES_BEFORE_4_GIB' if peak >= WATCHDOG else 'USER_INTERRUPT'
                    child.terminate()
                    break
                if psutil.disk_usage(bano.cohort.preflight.H).free < 10*1024**3:
                    reason = 'STOP_DISK_BELOW_10_GIB'
                    child.terminate()
                    break
                time.sleep(.05)
            try:
                rc = child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                rc = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    save(directory/'receipt.json', dict(exit_code=rc, owned_child_exit_confirmed=True,
        stop_reason=reason, peak_sampled_child_rss_bytes=peak, elapsed_seconds=time.monotonic()-started))
    print(str(directory), flush=True)
    if rc or reason:
        raise RuntimeError('STOP: '+str(directory/'worker.log'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run', choices=['bano'])
    modes.add_argument('--worker', choices=['bano'])
    p.add_argument('--directory', type=Path)
    a = p.parse_args()
    if a.worker:
        if a.directory is None or not a.directory.resolve().is_relative_to(OUT.resolve()):
            raise ValueError('New private v174 worker namespace required')
        bano_fidelity(a.directory)
    else:
        run(a.run)
