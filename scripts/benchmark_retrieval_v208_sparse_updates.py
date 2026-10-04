"""Fixed numeric sparse/dense storage benchmark; never business features or GT."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.xgb_matcher.retrieval_warm_sparse_names_v208 import WarmOrderedVector
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, quiet
from src.xgb_matcher.retrieval_v20_workbench import write_json


def run(directory):
    directory = Path(directory)
    allowed = (Path('/private/tmp'), Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v208_numeric_storage'))
    if not directory.is_dir() or any(directory.iterdir()) or not any(directory.resolve().is_relative_to(p) for p in allowed):
        raise ValueError('Fresh empty own numeric benchmark directory required')
    gate = quiet(30, 50, 55); write_json(directory / 'preflight.json', gate)
    rows = []
    with Budget(LIMITS) as budget:
        rng = np.random.default_rng(208); size = 30000000
        for density, count in (('dense', 1048576), ('sparse', 65536)):
            ids = rng.integers(0, size, count, dtype=np.uint32)
            scores = rng.integers(1, 44, count).astype(np.float64)/44
            for trial in range(3):
                outputs = {}
                modes = (('original', DiskVector), ('warm_ordered', WarmOrderedVector))
                for mode, cls in modes if trial%2 == 0 else reversed(modes):
                    began = time.perf_counter()
                    v = cls(directory / f'{density}_{trial}_{mode}.npy', size, np.float64)
                    init = time.perf_counter()-began; began = time.perf_counter()
                    for unused in range(4): v.maximum(ids, scores)
                    maximum = time.perf_counter()-began; outputs[mode] = v
                    rows.append(dict(density=density, trial=trial, mode=mode, initialization_seconds=init,
                        maximum_seconds=maximum, init_and_four_maximum_seconds=init+maximum,
                        references=4*count, world_positions=size,
                        profile=dict(v.update_profile) if mode != 'original' else None))
                for start in range(0, size, 131072):
                    stop = min(size, start+131072)
                    if outputs['original'].read(start, stop).tobytes() != outputs['warm_ordered'].read(start, stop).tobytes():
                        raise ValueError('NON_PARITY; no cost admitted')
                for v in outputs.values(): v.close(remove=True)  # These exact own numeric files only.
    result = dict(status='MEASURED_V208_NUMERIC_SPARSE_DENSE_STORAGE_NOT_NATIONAL', rows=rows,
        resources=budget.report(), every_world_vector_bit_equal=True, numeric_fixture_only=True,
        business_features_synthetic=False, alternating_order=True, initializer_cost_included=True,
        disk_cache_state='UNCONTROLLED', truth_opened=False, recall_measured=False,
        national_source_scope_measured=False, ready_for_fold1=False)
    result['storage_root'] = str(directory.resolve())
    result['same_filesystem_as_catnat'] = directory.stat().st_dev == Path('/Volumes/CATNAT_DATA').stat().st_dev
    write_json(directory / 'report.json', result); print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True, type=Path)
    run(p.parse_args().output)
