"""Execution-only continuation of predeclared V154; original immutable.

Memory128->64MiB, public-unit temporary table->identical projected view.
No parser/tokenizer/confidence/source/query/time/privacy/scientific change.
"""
import argparse
import hashlib
from pathlib import Path
import time

ORIGINAL = Path(__file__).with_name('embeddinggemma2_v154_direct_corporate_activity_source.py')
ORIGINAL_SHA = 'dcddcdc56b4a9e24c39d67a661b3bd32898d273f466826e3e30bb4fdd0f2a178'
text = ORIGINAL.read_text()
if hashlib.sha256(text.encode()).hexdigest() != ORIGINAL_SHA:
    raise ValueError('Original immutable V154 source drift')
body = text.split("\nif __name__=='__main__':",1)[0]
changes = {
    "SET memory_limit='128MB'":"SET memory_limit='64MB'",
    'CREATE TEMP TABLE public_units AS SELECT':'CREATE TEMP VIEW public_units AS SELECT',
    'DROP TABLE public_units':'DROP VIEW public_units',
    "'scripts.embeddinggemma2_v154_direct_corporate_activity_source','worker'":"'scripts.embeddinggemma2_v154_direct_corporate_activity_lowmem','worker'",
}
for before, after in changes.items():
    if body.count(before) != 1:
        raise ValueError('Execution substitution no longer unique')
    body = body.replace(before,after)
exec(compile(body,str(ORIGINAL),'exec'),globals())
OUT = io.HOME/'v154/direct_corporate_activity_20261014_night_b'
RECIPE = dict(RECIPE,duckdb_memory_limit='64MB',public_unit_representation='IDENTICAL_PROJECTED_VIEW_NO_MATERIALIZED_TEMP_TABLE',
    execution_only_continuation=True, original_source_implementation_sha256=ORIGINAL_SHA)
initialize_original = initialize


def initialize():
    initialize_original()
    prior = io.HOME/'v154/direct_corporate_activity_20261014_night'
    receipt = io.read(prior/'owned_exit.json')
    if receipt['exit_code'] == 0 or not receipt['own_child_exit_confirmed'] or (prior/'progress.json').exists():
        raise ValueError('Original stopped before any completed year; no duplicated active computation')
    launch = io.read(OUT/'launch.json')
    launch.update(original_source_implementation=record(ORIGINAL),
        original_failed_launch=record(prior/'launch.json'),original_actual_exit=record(prior/'owned_exit.json'),
        science_unchanged=True,no_completed_source_shards_migrated=True)
    io.write(OUT/'launch.json',launch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args()
    try:
        globals()[a.action]()
    except BaseException as error:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
