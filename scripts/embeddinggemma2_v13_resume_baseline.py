"""Owned restarts of unchanged V12 worker, same shards and scientific binding.

Only the exact resource-guard failure is retriable. Original receipts are
archived before overwrite, and no active process may be duplicated. Test stays
closed. Reader lifecycle resets by process exit, never by changing search.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v13'
MODULE = 'scripts.embeddinggemma2_v12_full_validation_lexical'
RESOURCE_ERROR = 'Real resource protection; shards retained for resume'


def retriable(code, new_log):
    return code != 0 and RESOURCE_ERROR in new_log


def assert_inactive(source):
    import psutil
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            args = proc.info['cmdline'] or []
            if MODULE in args and '--output' in args:
                index = args.index('--output') + 1
                if index < len(args) and Path(args[index]).resolve() == source.resolve():
                    raise RuntimeError(f'Existing worker/parent {proc.pid}; never duplicate')
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue


def count_shards(source, panel):
    total = 0
    for lo in range(0, len(panel), 16):
        seal = source / f'batch_{lo:04d}.seal.json'
        if not seal.exists(): break
        rows = io.read(verify(io.read(seal)))
        if [r['query_id'] for r in rows] != [q['query_id'] for q in panel[lo:lo+16]]:
            raise ValueError('Shard resume query binding')
        if any(len(r['candidates'])>100 or len({c['siret'] for c in r['candidates']})!=len(r['candidates']) for r in rows):
            raise ValueError('Shard initial100 cap')
        total += len(rows)
    return total


def run(source, out):
    assert_inactive(source)
    original = io.read(source / 'launch.json')
    for ref in original['implementation']: verify(ref)
    panel = io.read(verify(original['panel']))
    if len(panel)!=2212 or any(r['fold'] not in (2,3,4) for r in panel): raise ValueError('Unchanged full validation')
    out.mkdir(parents=True, exist_ok=False)
    binding = dict(source_launch=record(source/'launch.json'), panel=original['panel'], census=original['census'],
        implementation=record(Path(__file__)), science_unchanged=True, worker_unchanged=True,
        test_opened=False, folds01_closed=True)
    io.write(out/'launch.json', binding)
    for name in ('owned_exit.json','state.json'):
        if (source/name).exists(): io.write(out/('original_'+name),io.read(source/name))
    attempt=0; no_progress=0
    while True:
        verify(binding['source_launch']); verify(binding['implementation'])
        for ref in original['implementation']: verify(ref)
        free,level=io.pressure()
        if free<30 or level not in (0,1,2):
            io.write(out/'state.json',dict(status='WAITING_REAL_RESOURCE_HEADROOM',free_percent=free,pressure=level,attempts=attempt))
            time.sleep(30); continue
        assert_inactive(source)
        before=count_shards(source,panel)
        if (source/'report.json').exists() and io.read(source/'owned_exit.json')['exit_code']==0:
            io.write(out/'state.json',dict(status='COMPLETE_EXISTING_BASELINE',report=record(source/'report.json'))); return
        attempt+=1
        log=source/'generation.log'; offset=log.stat().st_size if log.exists() else 0
        command=[sys.executable,'-u','-m',MODULE,'run','--census',str(Path(original['census']['path']).parent),'--output',str(source)]
        io.write(out/'state.json',dict(status='OWNED_BASELINE_RESUME',attempt=attempt,completed_queries_before=before,total=2212,command=command))
        with (out/f'attempt_{attempt:02d}.log').open('x') as stdout:
            child=subprocess.Popen(command,stdout=stdout,stderr=subprocess.STDOUT)
            io.write(out/f'attempt_{attempt:02d}_process.json',dict(pid=child.pid,command=command))
            code=child.wait()
        after=count_shards(source,panel)
        with log.open('rb') as stream: stream.seek(offset); new_log=stream.read().decode('utf-8',errors='replace')
        io.write(out/f'attempt_{attempt:02d}_exit.json',dict(exit_code=code,own_child_exit_confirmed=True,
            command=command,before=before,after=after,ended_unix=time.time()))
        for name in ('owned_exit.json','state.json'):
            if (source/name).exists(): io.write(out/f'attempt_{attempt:02d}_{name}',io.read(source/name))
        if code==0:
            if after!=2212 or not (source/'report.json').exists(): raise ValueError('Complete generation/evaluation required')
            io.write(out/'state.json',dict(status='COMPLETE_RESUMED_FULL_VALIDATION_BASELINE',attempts=attempt,report=record(source/'report.json'))); return
        if not retriable(code,new_log):
            io.write(out/'state.json',dict(status='STOP_NON_RESOURCE_ERROR',attempt=attempt,exit_code=code))
            raise RuntimeError('Non-resource failure; preserved, no blind retry')
        no_progress = no_progress+1 if after<=before else 0
        if no_progress>=3:
            io.write(out/'state.json',dict(status='STOP_THREE_RESOURCE_EXITS_WITHOUT_PROGRESS',attempts=attempt,queries=after))
            raise RuntimeError('Three fresh workers made no progress; intervention required')
        io.write(out/'state.json',dict(status='RESOURCE_EXIT_SHARDS_PRESERVED_RESTARTING_READER',attempt=attempt,queries=after,no_progress=no_progress))
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT) or not args.source.resolve().is_relative_to(io.HOME/'v12'):
        raise ValueError('Dedicated V13 supervision / V12 unchanged baseline')
    run(args.source,args.output)
