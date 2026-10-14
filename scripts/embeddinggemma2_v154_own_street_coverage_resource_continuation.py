"""Execution-only continuation after observed V147 label-side RSS failure.

Matching/source/labels are unchanged. Typed scalar technical lookups avoid
uint64/int64 array promotion; release source mmaps before the stock lookup.
"""
from pathlib import Path
import argparse
import hashlib
import shutil
import time

import psutil

_ORIGINAL_SOURCE = Path(__file__).with_name('embeddinggemma2_v147_target_coverage_diagnostic.py')
_EXPECTED = 'dfece058847349272509d075df4793b30ab6fe7f0fa0fbe42dfd410a82e11eb7'
_ORIGINAL_OUT = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v147/target_coverage_diagnostic_20261014_night')
_RESOURCE_POLICY = dict(threads=1, max_rss_bytes=int(1.5 * 1024**3), min_free_percent=25,
    pressure_levels=[0, 1, 2], catnat_reserve_bytes=40 * 1024**3, deadline_unix=1791954368,
    system_resource_check_interval_seconds=5)
_text = _ORIGINAL_SOURCE.read_text()
if hashlib.sha256(_text.encode()).hexdigest() != _EXPECTED:
    raise ValueError('Original frozen failed audit implementation drift')
_body = _text.rsplit("\nif __name__ == '__main__':", 1)[0]
_changes = {
    "OUT = ROOT / 'v147/target_coverage_diagnostic_20261014_night'":
        "OUT = ROOT / 'v147/target_coverage_diagnostic_20261014_night_b'",
    'np.searchsorted(keys, target)': 'np.searchsorted(keys, np.uint64(target))',
    'np.searchsorted(universe, target)': 'np.searchsorted(universe, np.uint64(target))',
    '    postal = {}': '    for mm in (keys, xy, public, universe):\n        mm._mmap.close()\n    source.guard()\n    postal = {}',
    "'scripts.embeddinggemma2_v147_target_coverage_diagnostic'":
        "'scripts.embeddinggemma2_v154_own_street_coverage_resource_continuation'",
    "    write('recipe.json', RECIPE)": """    write('recipe.json', RECIPE)
    write('execution_continuation.json', dict(original_implementation=source.record(_ORIGINAL_SOURCE),
        original_failed_owned_exit=source.record(_ORIGINAL_OUT / 'owned_tool_exit.json'),
        original_completed_source_preflight=source.record(_ORIGINAL_OUT / 'source_preflight.json'),
        resource_policy=_RESOURCE_POLICY, parent_authorized_execution_only_1_5gib=True,
        changes=['TECHNICAL_LOOKUP_SCALARS_UINT64', 'RELEASE_MMAP_BEFORE_STOCK', 'EXECUTION_RSS_1_5GIB_WITH_SYSTEM_GUARDS'],
        source_matching_labels_denominators_and_helpers_unchanged=True))""",
}
for _before, _after in _changes.items():
    if _body.count(_before) != 1:
        raise ValueError('Exact execution-only transformation required')
    _body = _body.replace(_before, _after)
exec(compile(_body, str(_ORIGINAL_SOURCE), 'exec'), globals())
RECIPE.update(max_rss_bytes=_RESOURCE_POLICY['max_rss_bytes'], execution_policy=_RESOURCE_POLICY,
              execution_only_continuation_no_scientific_change=True)


_last_system_check = 0.
_last_system_snapshot = None


def _execution_guard():
    global _last_system_check, _last_system_snapshot
    if time.time() >= _RESOURCE_POLICY['deadline_unix']:
        raise TimeoutError('Absolute night deadline')
    rss = psutil.Process().memory_info().rss
    # Original shared utility is imported explicitly, never modified on disk.
    from scripts import embeddinggemma2_night as io
    now = time.monotonic()
    if _last_system_snapshot is None or now - _last_system_check >= 5:
        free, pressure = io.pressure()
        _last_system_snapshot = (free, pressure, shutil.disk_usage(ROOT).free)
        _last_system_check = now
    free, pressure, disk_free = _last_system_snapshot
    if rss > _RESOURCE_POLICY['max_rss_bytes'] or free < 25 or pressure not in (0, 1, 2):
        raise RuntimeError(f'Execution-only resource guard rss={rss} free={free} pressure={pressure}')
    if disk_free < _RESOURCE_POLICY['catnat_reserve_bytes']:
        raise RuntimeError('CATNAT40GiB reserve')
    return rss


source.guard = _execution_guard  # Process-local execution policy only.


if __name__ == '__main__':
    failed = source.prior.read(_ORIGINAL_OUT / 'owned_tool_exit.json')
    if failed['exit_code'] != 1 or not failed['own_child_exit_confirmed']:
        raise ValueError('Actual observed failed audit before continuation')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'audit'))
    args = parser.parse_args()
    globals()[args.action]()
