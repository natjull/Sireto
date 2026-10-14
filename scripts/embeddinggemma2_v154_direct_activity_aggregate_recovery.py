"""Resume19 completed source years byte-identically; aggregate memory1GiB.

No source parsing again, no new formula/query/role/cutoff; RSS2GiB monitored.
Original source DB/checkpoints remain intact. Only confidence/provenance export
is resumed under a sufficient DuckDB execution limit.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import time

ORIGINAL=Path(__file__).with_name('embeddinggemma2_v154_direct_corporate_activity_source.py')
ORIGINAL_SHA='dcddcdc56b4a9e24c39d67a661b3bd32898d273f466826e3e30bb4fdd0f2a178'
text=ORIGINAL.read_text()
if hashlib.sha256(text.encode()).hexdigest()!=ORIGINAL_SHA:
    raise ValueError('Frozen V154 algorithm drift')
body=text.split("\nif __name__=='__main__':",1)[0]
changes={"SET memory_limit='128MB'":"SET memory_limit='1GB'",
    "'scripts.embeddinggemma2_v154_direct_corporate_activity_source','worker'":"'scripts.embeddinggemma2_v154_direct_activity_aggregate_recovery','worker'"}
for before,after in changes.items():
    if body.count(before)!=1:raise ValueError('Unique execution substitution')
    body=body.replace(before,after)
before="""        guard(); verify(ref['manifest']); source = verify(ref['source'])
        if str(source) in progress['years']:
            continue"""
after="""        guard(); verify(ref['manifest']); source = Path(ref['source']['path'])
        if str(source) in progress['years']:
            if source.stat().st_size != ref['source']['size_bytes']:
                raise ValueError('Inherited already-verified source size drift')
            continue
        source = verify(ref['source'])"""
if body.count(before)!=1:raise ValueError('Exact completed-year execution skip')
body=body.replace(before,after)
exec(compile(body,str(ORIGINAL),'exec'),globals())
OUT=io.HOME/'v154/direct_corporate_activity_20261014_night_d'
PRIOR_COMPLETED=io.HOME/'v154/direct_corporate_activity_20261014_night_c'
RECIPE=dict(RECIPE,duckdb_memory_limit='1GB',rss_max_bytes=2*1024**3,execution_only_recovery=True,
    original_source_implementation_sha256=ORIGINAL_SHA,
    completed19years_source_hashes_inherited_from_pre_and_post_parse_verified_authority=True,
    no_source_rows_reparsed=True,no_scientific_change=True)
initialize_original=initialize


def initialize():
    initialize_original()
    ex=io.read(PRIOR_COMPLETED/'owned_exit.json'); progress=io.read(PRIOR_COMPLETED/'progress.json')
    if ex['exit_code']==0 or not ex['own_child_exit_confirmed'] or len(progress['years'])!=19:
        raise ValueError('Actual stopped completed19-year checkpoint required')
    old=io.read(PRIOR_COMPLETED/'launch.json'); launch=io.read(OUT/'launch.json')
    for key in ('query_panel','sources','public_units','tokenizer','source_authority','source_actual_exit'):
        if launch[key]!=old[key]:raise ValueError('Exact migration binding drift: '+key)
    copies=[]
    for filename in ('aggregate.duckdb','aggregate.duckdb.wal','progress.json'):
        src=PRIOR_COMPLETED/filename
        if not src.exists():continue
        before=record(src);shutil.copy2(src,OUT/filename);after=record(OUT/filename)
        if (before['sha256'],before['size_bytes'])!=(after['sha256'],after['size_bytes']):
            raise ValueError('Byte-identical checkpoint migration failed')
        copies.append(dict(source=before,destination=after))
    io.write(OUT/'migration.json',dict(status='COMPLETE_BYTE_IDENTICAL19YEAR_AGGREGATE_MIGRATION',
        copies=copies,original_launch=record(PRIOR_COMPLETED/'launch.json'),actual_exit=record(PRIOR_COMPLETED/'owned_exit.json'),
        original_implementation=record(ORIGINAL),original_parser_and_source_hashes_unchanged=True,
        no_scientific_or_source_or_query_or_formula_change=True))
    launch.update(migration=record(OUT/'migration.json'),no_completed_year_reparse=True,science_unchanged=True)
    io.write(OUT/'launch.json',launch)


def run():
    command=[sys.executable,'-u','-m','scripts.embeddinggemma2_v154_direct_activity_aggregate_recovery','worker']
    peak=0;stopped=None
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
    if rc:raise RuntimeError(f'Whole source aggregation exit{rc}; no partial admission')
    print('OWNED_SOURCE_EXIT0',child.pid,peak,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException as error:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
