"""Brief10 serial fidelity controller: 6GiB admitted at7.5GiB free+inactive,
8GiB if hydraulic is stopped. Read-only VM probes; no full3000 launch mode.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import psutil
# Reuse the already installed streaming parser with the SAME Python3.14 ABI.
# Import it alone from the CPU environment, then restore the Metal environment's
# package precedence (notably torch/numpy/transformers). No dependency installed.
try:
    import ijson
except ModuleNotFoundError:
    parser_site=ROOT/'.venv/lib'/('python'+'.'.join(map(str,sys.version_info[:2])))/'site-packages'
    sys.path.insert(0,str(parser_site))
    try:import ijson
    finally:sys.path.remove(str(parser_site))
from scripts import train3000_frozen_v142_fidelity as mini
from scripts import retrieval_site_target_full as resources

GIB=1024**3
OUT=mini.bano.cohort.OUT/'brief10'
CAP=int(os.environ.get('SIRETO_BRIEF10_CAP_BYTES',6*GIB))
MARGIN=256*1024**2
STOP=False
LAST_TELEMETRY=0.


def guard():
    global LAST_TELEMETRY
    rss=psutil.Process().memory_info().rss
    torch=sys.modules.get('torch');metal=0
    if torch is not None and torch.backends.mps.is_available():
        metal=int(torch.mps.driver_allocated_memory())
    # Conservative unified-memory accounting: Metal may overlap process RSS.
    total=rss+metal
    target=os.environ.get('SIRETO_BRIEF10_TELEMETRY')
    if target and time.monotonic()-LAST_TELEMETRY>.5:
        with Path(target).open('a') as stream:
            stream.write(json.dumps(dict(unix=time.time(),rss_bytes=rss,metal_driver_bytes=metal,
                conservative_total_bytes=total))+'\n')
        LAST_TELEMETRY=time.monotonic()
    if total>=CAP-MARGIN:
        raise MemoryError('WAIT_BRIEF10_RESOURCE_CAP')
    if psutil.disk_usage(mini.bano.cohort.preflight.H).free<10*GIB:
        raise RuntimeError('STOP_DISK_BELOW_10_GIB')
    return rss


mini.guard=guard
mini.LIMIT=CAP
mini.WATCHDOG=CAP-MARGIN


def interrupted(*args):
    global STOP
    STOP=True


def publish(path,value):
    mini.save(path,value)


def admission(directory,minimum_cap):
    waited=0
    while True:
        if STOP:raise InterruptedError('Owned controller interrupted, no VM action')
        vm=resources.hydraulic();memory=resources.free_inactive_memory()
        cap=(8 if vm['status']=='Stopped' and not vm.get('uncertain') else 6)*GIB
        free=memory['free_inactive_bytes']
        ready=not vm.get('uncertain') and free is not None and free>=7.5*GIB and cap>=minimum_cap
        row=dict(unix=time.time(),hydraulic=vm,memory=memory,rss_cap_bytes=cap,
            required_cap_bytes=minimum_cap,required_free_inactive_bytes=int(7.5*GIB),admitted=ready,
            previous_wait_seconds=waited,vm_action='NONE')
        with (directory/'admission.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(dict(phase='ADMITTED' if ready else 'WAIT_MEMORY_60S',family_directory=str(directory),
            cap_bytes=cap,free_inactive_bytes=free,minimum_cap=minimum_cap,wait_seconds=waited)),flush=True)
        if ready:return cap,waited,row
        for _ in range(60):
            if STOP:raise InterruptedError('Interrupted while waiting; no worker')
            time.sleep(1)
        waited+=60


def latest_pass(family):
    for directory in sorted(OUT.glob(family+'_*'),reverse=True):
        if not (directory/'result.json').exists() or not (directory/'receipt.json').exists():continue
        result=mini.read(directory/'result.json');receipt=mini.read(directory/'receipt.json')
        if result['status']=='PASS' and receipt['exit_code']==0 and receipt['owned_child_exit_confirmed']:
            for ref in mini.read(directory/'controller.json')['sources']:
                mini.record(directory/Path(ref['path']).name,ref['sha256'])
                implementation='numeric' if family=='numeric51' else family
                if Path(ref['path']).name in ('train3000_frozen_v142_'+implementation+'.py',
                        'train3000_frozen_v142_fidelity.py','train3000_frozen_v142_bano.py'):
                    mini.record(ref['path'],ref['sha256'])
            return dict(result=result,result_ref=mini.record(directory/'result.json'),receipt_ref=mini.record(directory/'receipt.json'))
    return None


def stage(family):
    previous=latest_pass(family)
    if previous:return previous
    if any(p.exists() for p in OUT.glob('*/failure.json')):
        raise RuntimeError('STOP_PRIOR_FIDELITY_FAILURE_PRESERVED')
    minimum_cap=6*GIB
    for prior in OUT.glob(family+'_*/receipt.json'):
        receipt=mini.read(prior)
        if receipt['stop_reason']=='WAIT_BRIEF10_RESOURCE_CAP':
            cap=receipt.get('cap_bytes',mini.read(prior.parent/'controller.json')['cap_bytes'])
            if cap>=8*GIB:raise RuntimeError('Terminal8GiB resource stop; preserve all attempts')
            minimum_cap=8*GIB
    while True:
        directory=OUT/(family+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);directory.mkdir()
        cap,waited,evidence=admission(directory,minimum_cap)
        interpreter=mini.bano.cohort.preflight.H/'runtime/bin/python' if family=='numeric51' else ROOT/'.venv/bin/python'
        command=[str(interpreter),'-u',str(Path(__file__).resolve()),'--worker',family,'--directory',str(directory)]
        env=dict(os.environ,SIRETO_BRIEF10_CAP_BYTES=str(cap),SIRETO_BRIEF10_TELEMETRY=str(directory/'memory.jsonl'),
            PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        tmp=directory/'tmp';tmp.mkdir()
        env.update(TMPDIR=str(tmp),SQLITE_TMPDIR=str(tmp),OSMIUM_POOL_THREADS='1')
        refs=[]
        for name in ('brief10','numeric','osm','activity','fidelity','bano'):
            path=ROOT/'scripts'/('train3000_frozen_v142_'+name+'.py')
            ref=mini.record(path);archive=directory/path.name
            with archive.open('xb') as stream:stream.write(path.read_bytes())
            mini.record(archive,ref['sha256']);refs.append(ref)
        started=time.monotonic();peak=0;reason=None
        with (directory/'worker.log').open('xb') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            observed_vm=[evidence['hydraulic']]
            def watch_vm(owned_child=child,observed_state=observed_vm):
                while owned_child.poll() is None:
                    observed_state[0]=resources.hydraulic()
                    time.sleep(2)
            threading.Thread(target=watch_vm,daemon=True).start()
            publish(directory/'controller.json',dict(pid=os.getpid(),child_pid=child.pid,command=command,
                sources=refs,cap_bytes=cap,watchdog_margin_bytes=MARGIN,admission=evidence,
                wait_seconds=waited,hydraulic_action='NONE'))
            try:
                while child.poll() is None:
                    try:
                        process=psutil.Process(child.pid)
                        rss=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running())
                        peak=max(peak,rss)
                    except psutil.NoSuchProcess:pass
                    if STOP or peak>=cap-MARGIN:
                        reason='USER_INTERRUPT' if STOP else 'WAIT_BRIEF10_RESOURCE_CAP'
                        child.terminate();break
                    if cap==8*GIB and (observed_vm[0]['status']!='Stopped' or observed_vm[0].get('uncertain')):
                        reason='WAIT_VM_RE_ADMISSION';child.terminate();break
                    if psutil.disk_usage(mini.bano.cohort.preflight.H).free<10*GIB:
                        reason='STOP_DISK_BELOW_10_GIB';child.terminate();break
                    time.sleep(.05)
                try:rc=child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();rc=child.wait()
            finally:
                if child.poll() is None:child.terminate();child.wait()
        # An in-worker resource guard is a resource wait, not non-fidelity.
        if (directory/'resource_yield.json').exists():reason='WAIT_BRIEF10_RESOURCE_CAP'
        publish(directory/'receipt.json',dict(exit_code=rc,owned_child_exit_confirmed=True,stop_reason=reason,
            peak_sampled_child_rss_bytes=peak,cap_bytes=cap,elapsed_seconds=time.monotonic()-started,wait_seconds=waited))
        print(json.dumps(dict(phase='EXIT',family=family,directory=str(directory),exit_code=rc,reason=reason)),flush=True)
        if STOP:raise InterruptedError('Owned child exited; outputs preserved')
        if reason=='WAIT_BRIEF10_RESOURCE_CAP' and cap<8*GIB:
            minimum_cap=8*GIB
            continue  # Wait indefinitely for Stopped AND >=7.5GiB, no futile6GiB retry.
        if reason=='WAIT_VM_RE_ADMISSION':
            continue
        if reason or rc:raise RuntimeError('STOP: '+str(directory/'worker.log'))
        return latest_pass(family)


def run(mode):
    OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (OUT/'owned.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        families=('activity','osm','numeric51') if mode=='fidelity' else (mode,)
        results={}
        for family in families:results[family]=stage(family)
        if mode=='fidelity':publish(OUT/'fidelity.json',dict(status='PASS_THREE_REMAINING_FAMILIES',families=results,
            prior_bano_proof=mini.record(ROOT/'docs/train3000_frozen_v142_bano_fidelity_brief9.json'),
            native_smoke_queries=0,full3000_started=False))


def worker(family,directory):
    if directory is None or not directory.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Owned fresh brief10 v174 directory required')
    guard()
    import duckdb
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    connect=duckdb.connect
    class Connection:
        def __init__(self,*args,**kwargs):
            self.c=connect(*args,**kwargs);self.c.execute('SET threads=1')
        def __getattr__(self,name):return getattr(self.c,name)
        def __enter__(self):self.c.__enter__();return self
        def __exit__(self,*args):return self.c.__exit__(*args)
        def execute(self,sql,*args,**kwargs):
            return self.c.execute(re.sub(r'(?i)\bSET\s+threads\s*=\s*\d+\b','SET threads=1',sql),*args,**kwargs)
    duckdb.connect=Connection
    try:
        if family=='activity':
            from scripts import train3000_frozen_v142_activity as adapter
        elif family=='osm':
            from scripts import train3000_frozen_v142_osm as adapter
        else:
            from scripts import train3000_frozen_v142_numeric as adapter
        adapter.fidelity(directory)
    except MemoryError as error:
        publish(directory/'resource_yield.json',dict(reason=str(error),cap_bytes=CAP,vm_action='NONE'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run',choices=['fidelity','activity','osm','numeric51'])
    modes.add_argument('--worker',choices=['activity','osm','numeric51'])
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args.worker,args.directory)
    else:run(args.run)
