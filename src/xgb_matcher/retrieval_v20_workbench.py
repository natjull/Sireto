"""Versioned v20 diagnostics; no changes to the historical retrieval engines."""
from __future__ import annotations

import _thread
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

from .retrieval_parent_discovery_v18 import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def verify(record):
    p = Path(record['path'])
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file() or sha256_file(p) != record['sha256']:
        raise ValueError(f'Authority drift: {p}')
    return p


def load_config(path):
    c = json.loads(Path(path).read_text())
    if c['schema'] != 'sireto-v20-evidence-workbench-1':
        raise ValueError('Config schema')
    return c, {k: json.loads(verify(v).read_text()) for k, v in c['source_configs'].items()}


def record(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'sha256': sha256_file(p), 'size_bytes': p.stat().st_size}


def implementation(script):
    paths = [Path(script),Path(__file__),ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',
      ROOT/'src/xgb_matcher/retrieval_parent_fusion_v19.py',ROOT/'src/xgb_matcher/retrieval_geo_v16.py',
      ROOT/'src/xgb_matcher/retrieval_site_inventory_v17.py',ROOT/'src/xgb_matcher/retrieval_v12_scene_ir.py']
    return {str(p.relative_to(ROOT)):record(p) for p in paths}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def sql_path(path):
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def inner_validation(siren):
    return int(hashlib.sha256(('v20-inner|' + str(siren)).encode()).hexdigest(), 16) % 5 == 0


def negative_positions(count, method):
    """Indices into naturally present negatives, already in deterministic RRF order."""
    if method == 'HISTORICAL_HEAD512':
        return np.arange(min(count, 512), dtype=np.int64)
    if method != 'HEAD256_PLUS_STRATIFIED_TAIL256':
        raise ValueError('Unknown negative sampler')
    if count <= 512:
        return np.arange(count, dtype=np.int64)
    # Exactly one representative per equal-width stratum of the tail.
    tail = 256 + np.floor((np.arange(256) + .5) * (count - 256) / 256).astype(np.int64)
    return np.concatenate([np.arange(256), tail])


def choose_master(candidates, cap=2000, prefix=1750):
    """Reference implementation, also used to validate the SQL selection in tests."""
    if not 0 <= prefix <= cap <= 2000:
        raise ValueError('Invalid Master cap')
    unique = {r['siret']: r for r in candidates}
    if len(unique) != len(candidates):
        raise ValueError('Duplicate SIRET')
    base = sorted([r for r in candidates if r.get('base_rank') is not None and r['base_rank'] <= prefix], key=lambda r: r['base_rank'])
    if len(base) > prefix:
        raise ValueError('Invalid baseline ranks')
    keep = {r['siret'] for r in base}
    rescue = sorted([r for r in candidates if r['siret'] not in keep], key=lambda r: (-r['rescue_score'], -r['direct_rrf'], -r['parent_rrf'], r['siret']))
    return base + rescue[:cap-len(base)]


class Budget:
    """Monitor actual process RSS and machine swapouts, including native fits."""
    def __init__(self, config):
        self.c = config
        self.stop = threading.Event()
        self.failure = None
        self.peak = 0

    @staticmethod
    def swap():
        for line in subprocess.check_output(['/usr/bin/vm_stat'], text=True).splitlines():
            if line.startswith('Swapouts:'):
                return int(line.split(':')[1].strip().rstrip('.'))
        raise RuntimeError('Missing swap counter')

    def __enter__(self):
        self.start = time.monotonic()
        self.before = self.swap()
        self.after = self.before
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def _watch(self):
        while not self.stop.wait(1):
            try:
                self.peak = max(self.peak, psutil.Process().memory_info().rss)
                self.after = self.swap()
                if self.peak > self.c['max_rss_bytes'] or self.after-self.before > self.c['max_swapout_pages_delta'] or time.monotonic()-self.start > self.c['max_seconds']:
                    raise RuntimeError('STOP_RESOURCE_POLICY')
            except Exception as exc:
                self.failure = str(exc)
                _thread.interrupt_main()
                return

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=3)
        self.after = self.swap()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak = max(self.peak, int(peak if sys.platform == 'darwin' else peak*1024))
        self.elapsed = time.monotonic()-self.start
        if self.peak > self.c['max_rss_bytes'] or self.after-self.before > self.c['max_swapout_pages_delta'] or self.elapsed > self.c['max_seconds']:
            self.failure = self.failure or 'STOP_RESOURCE_POLICY'
        if self.failure:
            raise RuntimeError(self.failure)

    def report(self):
        return dict(wall_seconds=self.elapsed, peak_rss_bytes=self.peak, swapout_pages_delta=self.after-self.before, pass_resource_policy=self.failure is None)
