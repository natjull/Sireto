"""Synthetic arithmetic cost only, NOT national retrieval or business features."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.xgb_matcher.retrieval_grouped_proof_updates_v204 import GroupedProofVector
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, quiet
from src.xgb_matcher.retrieval_v20_workbench import write_json


def run(directory):
    directory = Path(directory)
    if not directory.is_dir() or any(directory.iterdir()) or not directory.resolve().is_relative_to(Path('/private/tmp')):
        raise ValueError('Fresh empty own private/tmp benchmark directory required')
    gate = quiet(30, 50, 55); write_json(directory / 'preflight.json', gate)
    rows = []; owned = []
    with Budget(LIMITS) as budget:
        rng = np.random.default_rng(204); size = 16000000
        ids = rng.integers(0, size, 1048576, dtype=np.uint32)
        scores = rng.integers(1, 44, len(ids)).astype(np.float64)/44
        flags = np.zeros(len(ids), dtype=np.bool_)
        for trial in range(3):
            outputs = {}
            for mode, cls in (('original', DiskVector), ('grouped', GroupedProofVector)) if trial%2 == 0 else (
                    ('grouped', GroupedProofVector), ('original', DiskVector)):
                cov = cls(directory / f'{trial}_{mode}_cov.npy', size, np.float64)
                exact = cls(directory / f'{trial}_{mode}_exact.npy', size, np.bool_)
                owned.extend((cov, exact)); started = time.perf_counter()
                for _ in range(4): cov.maximum(ids, scores)
                maximum_seconds = time.perf_counter()-started; started = time.perf_counter()
                for _ in range(4): exact.flags(ids, flags)
                rows.append(dict(trial=trial, mode=mode, maximum_seconds=maximum_seconds,
                    all_false_flags_seconds=time.perf_counter()-started))
                outputs[mode] = (cov, exact)
            for a, b in zip(outputs['original'], outputs['grouped']):
                for start in range(0, size, 131072):
                    end = min(size, start+131072)
                    if a.read(start, end).tobytes() != b.read(start, end).tobytes():
                        raise ValueError('NON_PARITY; no cost result admitted')
            for pair in outputs.values():
                for v in pair:
                    v.close(remove=True); owned.remove(v)  # Exact own numeric scratch files only.
    result = dict(status='MEASURED_V204_SYNTHETIC_OWN_VECTOR_COST_NOT_NATIONAL', rows=rows,
        resources=budget.report(), numeric_fixture_only=True, business_features_synthetic=False,
        every_world_vector_bit_equal=True, alternating_order=True, disk_cache_state='UNCONTROLLED',
        truth_opened=False, recall_measured=False, national_source_scope_measured=False, ready_for_fold1=False)
    write_json(directory / 'report.json', result); print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
