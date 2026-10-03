"""Read-only system stability gate, never weakening the worker swap policy."""
import re
import subprocess
import time

from src.xgb_matcher.retrieval_v20_workbench import Budget


class QuietWindow:
    def __init__(self,seconds,min_free_percent):
        if (not isinstance(seconds,(int,float)) or isinstance(seconds,bool) or not 0<seconds<=55
            or not isinstance(min_free_percent,int) or isinstance(min_free_percent,bool) or not 0<=min_free_percent<=100):
            raise ValueError('Bounded explicit system quiet window required')
        self.seconds=seconds;self.minimum=min_free_percent;self.last=None;self.start=None

    def observe(self,now,swapouts,free_percent):
        if (not isinstance(swapouts,int) or isinstance(swapouts,bool) or swapouts<0
            or not isinstance(free_percent,int) or isinstance(free_percent,bool) or not 0<=free_percent<=100):
            raise ValueError('Explicit system counters required')
        if self.last is not None and swapouts<self.last:raise ValueError('System swap counter reset, no stability inference')
        if self.last!=swapouts or free_percent<self.minimum:self.start=None
        self.last=swapouts
        if free_percent>=self.minimum and self.start is None:self.start=now
        return self.start is not None and now-self.start>=self.seconds


def snapshot():
    text=subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True,timeout=2)
    match=re.search(r'System-wide memory free percentage:\s*(\d+)%',text)
    if match is None:raise RuntimeError('Missing actual system memory-pressure counter')
    return dict(swapouts=Budget.swap(),free_percent=int(match.group(1)))


def quiet(seconds,min_free_percent,max_wait_seconds=55):
    if not isinstance(max_wait_seconds,(int,float)) or isinstance(max_wait_seconds,bool) or not seconds<=max_wait_seconds<=55:
        raise ValueError('At most55s bounded system preflight required')
    window=QuietWindow(seconds,min_free_percent);began=time.monotonic();samples=[]
    while True:
        s=snapshot();now=time.monotonic();samples.append(dict(elapsed_seconds=now-began,**s))
        if window.observe(now,**s):
            return dict(pass_system_quiet_policy=True,required_quiet_seconds=seconds,quiet_seconds=now-window.start,
                min_free_percent=min_free_percent,elapsed_seconds=now-began,final_snapshot=s,samples=samples,
                global_swap_delta_since_first_sample=s['swapouts']-samples[0]['swapouts'],
                global_swap_delta_in_successful_quiet_window=0,worker_swap_budget_changed=False)
        if now-began>=max_wait_seconds:raise RuntimeError('STOP_SYSTEM_NOT_QUIET_NO_WORKER_LAUNCHED')
        time.sleep(min(1,max_wait_seconds-(now-began)))
