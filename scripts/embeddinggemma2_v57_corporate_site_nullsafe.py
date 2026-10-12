"""V56 continuation with null-safe public address suffixes, before outputs.

The historical generic normalizer renders None as NONE. Missing repetition
suffix is absence, not that literal token. No coefficient or outcome selected.
The original V56 code/config remain immutable, in their stopped namespace.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts import embeddinggemma2_v56_corporate_site as experiment
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record

OUTPUT=io.HOME/'v57/corporate_site_nullsafe_20261013_a'
NORMALIZE=experiment.original.structural.normalize


def normalize(value,street=False):
    return NORMALIZE('' if value is None else value,street=street)


def initialize(out):
    experiment.initialize(out)
    recipe=io.read(out/'recipe.json')
    recipe.update(public_address_nulls='ABSENCE_NOT_LITERAL_NONE',
        continuation='NO_PRIOR_CANDIDATE_OUTPUTS_OR_LABELS_OPENED',
        wrapper_implementation=record(Path(__file__)))
    io.write(out/'recipe.json',recipe)
    launch=io.read(out/'launch.json')
    launch.update(recipe=record(out/'recipe.json'))
    launch['helpers'].append(record(Path(__file__)))
    io.write(out/'launch.json',launch)


def run(out):
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for action,artifact in [('prepare','features_generation.json'),('calibrate','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned completion required')
            continue
        while True:
            experiment.guard(experiment.DEADLINE)
            command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v57_corporate_site_nullsafe',action,'--output',str(out)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
            with (out/(action+'.log')).open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                io.write(out/(action+'_process.json'),dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                while child.poll() is None:
                    if engine.INTERRUPTED or time.time()>=experiment.DEADLINE-30:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(5)
                rc=child.wait()
            io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
            if engine.INTERRUPTED or time.time()>=experiment.DEADLINE-30:raise TimeoutError('EIGHT_HOUR_END_ARTIFACTS_PRESERVED')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} owned exit {rc}; no partial recall')
            break
    experiment.stage('evaluate',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','prepare','calibrate','generate','evaluate'])
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    experiment.original.structural.normalize=normalize
    try:
        if a.action=='initialize':initialize(a.output)
        elif a.action=='run':run(a.output)
        else:experiment.stage(a.action,a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
