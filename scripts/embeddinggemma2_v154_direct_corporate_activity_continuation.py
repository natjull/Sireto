"""Execution-only RSS2GiB continuation; frozen V154 science and disk backing.

Original128MiB DuckDB/table execution retained. Parents monitor true worker
RSS continuously; large source confidence/provenance remain disk-backed.
"""
import argparse
import hashlib
from pathlib import Path
import time

ORIGINAL = Path(__file__).with_name('embeddinggemma2_v154_direct_corporate_activity_source.py')
ORIGINAL_SHA = 'dcddcdc56b4a9e24c39d67a661b3bd32898d273f466826e3e30bb4fdd0f2a178'
body = ORIGINAL.read_text()
if hashlib.sha256(body.encode()).hexdigest() != ORIGINAL_SHA:
    raise ValueError('Frozen V154 parser/algorithm drift')
body = body.split("\nif __name__=='__main__':",1)[0]
old="'scripts.embeddinggemma2_v154_direct_corporate_activity_source','worker'"
if body.count(old) != 1:
    raise ValueError('Unique execution module binding')
body=body.replace(old,"'scripts.embeddinggemma2_v154_direct_corporate_activity_continuation','worker'")
exec(compile(body,str(ORIGINAL),'exec'),globals())
OUT=io.HOME/'v154/direct_corporate_activity_20261014_night_c'
RECIPE=dict(RECIPE,rss_max_bytes=2*1024**3,execution_only_continuation=True,
    original_source_implementation_sha256=ORIGINAL_SHA,
    resource_authority='ROOT_EXPLICIT_1_TO_2GIB_EXECUTION_ONLY_WITH_FREE25_PRESSURE012_DISK40',
    aggregate_representation='DISK_BACKED_ALL_OWNERS_TOKEN_OWNER_PUBLICATION_AND_QUERY_CONFIDENCE')
initialize_original=initialize


def initialize():
    initialize_original()
    prior=io.HOME/'v154/direct_corporate_activity_20261014_night'
    ex=io.read(prior/'owned_exit.json')
    if ex['exit_code']==0 or not ex['own_child_exit_confirmed'] or (prior/'progress.json').exists():
        raise ValueError('No active source/complete original years duplicated')
    launch=io.read(OUT/'launch.json')
    launch.update(original_source_implementation=record(ORIGINAL),original_failed_exit=record(prior/'owned_exit.json'),
        science_unchanged=True,prior_failed_view_namespace_preserved=True,no_completed_years_migrated=True)
    io.write(OUT/'launch.json',launch)


def run():
    command=[sys.executable,'-u','-m','scripts.embeddinggemma2_v154_direct_corporate_activity_continuation','worker']
    peak=0; stopped=None
    with (OUT/'worker.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(OUT/'spill')))
        io.write(OUT/'owned_process.json',dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
        while child.poll() is None:
            try:rss=psutil.Process(child.pid).memory_info().rss;peak=max(peak,rss)
            except psutil.NoSuchProcess:break
            free,level=io.pressure();disk=psutil.disk_usage(str(io.HOME)).free
            if time.time()>=DEADLINE-20 or rss>RECIPE['rss_max_bytes'] or free<25 or level not in (0,1,2) or disk<40*1024**3:
                stopped=dict(rss=rss,free=free,level=level,disk=disk,deadline=time.time()>=DEADLINE-20)
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(3)
        rc=child.wait()
    io.write(OUT/'owned_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,
        observed_worker_peak_rss_bytes=peak,stopped_on_actual_guard=stopped,finished_unix=time.time()))
    if rc:
        raise RuntimeError(f'Whole-source worker exit{rc}; no partial source admitted')
    print('OWNED_SOURCE_EXIT0',child.pid,peak,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException as error:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
