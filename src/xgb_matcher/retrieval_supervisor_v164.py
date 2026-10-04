"""V164 race-safe supervision: unchanged limits, measured STOP diagnostics.

NoSuchProcess can occur after poll(None) while snapshot() is running. Only
wait(0) of our owned Popen child confirms success; disappearance alone does
not. Nonzero exit, timeout, pressure/RSS/counter failure remain fatal.
"""
import subprocess
import time

import psutil

from .retrieval_memory_policy_v153 import LIMITS,POLICY,check_sample,snapshot
from .retrieval_v20_workbench import write_json


def supervise(command,log,env,stop_path):
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    began=time.monotonic();previous=None;peak=0;minimum=100;samples=0;disappeared=False
    first=None;last_sample=None
    try:
        while child.poll() is None:
            sample=snapshot();last_sample=sample
            minimum=min(minimum,sample['free_percent']);samples+=1
            check_sample(sample,previous,peak,time.monotonic()-began,LIMITS)
            previous=sample['swapouts']
            if first is None:first=previous
            try:rss=psutil.Process(child.pid).memory_info().rss
            except psutil.NoSuchProcess:
                # Never infer a successful exit from a missing PID. Reap our
                # exact child and inspect its actual return code below.
                disappeared=True;child.wait(timeout=5);break
            peak=max(peak,rss)
            check_sample(sample,previous,peak,time.monotonic()-began,LIMITS)
            time.sleep(.5)
        code=child.wait(timeout=5)
        if code:raise subprocess.CalledProcessError(code,command)
        final=snapshot();last_sample=final;minimum=min(minimum,final['free_percent']);samples+=1
        check_sample(final,previous,peak,time.monotonic()-began,LIMITS)
        return dict(policy_id=POLICY,limits=LIMITS,child_pid=child.pid,exit_code=code,
            own_child_exit_confirmed=True,child_disappeared_during_sampling=disappeared,
            peak_rss_bytes=peak,minimum_system_free_percent=minimum,sample_count=samples,
            wall_seconds=time.monotonic()-began,global_swap_is_worker_attributed=False,
            swapout_pages_delta=final['swapouts']-(first if first is not None else final['swapouts']))
    except BaseException as exc:
        failure_elapsed=time.monotonic()-began
        cleanup_error=None
        try:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        except Exception as cleanup:
            cleanup_error=str(cleanup)
        write_json(stop_path,dict(status='STOP',reason=str(exc),policy_id=POLICY,limits=LIMITS,
            child_pid=child.pid,exit_code=child.returncode,own_child_exit_confirmed=child.returncode is not None,
            child_disappeared_during_sampling=disappeared,peak_rss_bytes=peak,
            minimum_system_free_percent=minimum,sample_count=samples,last_sample=last_sample,
            cleanup_error=cleanup_error,
            failure_wall_seconds=failure_elapsed,wall_seconds=time.monotonic()-began,
            global_swap_is_worker_attributed=False,swapout_pages_delta=(previous-first)
            if first is not None and previous is not None else None))
        raise
