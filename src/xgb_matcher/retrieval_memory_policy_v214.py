"""Authorized V214 memory policy. Global swap is telemetry, not attribution.

Only macOS normal pressure (kernel level 1), >=50% memory_pressure free,
and <=2GiB RSS are accepted. Counters failing/resetting stop the operation.
The parent independently supervises only its own child; no system mutation.
"""
import _thread
import resource
import subprocess
import sys
import threading
import time

import psutil

from src.xgb_matcher.retrieval_system_quiet_v143 import snapshot as legacy_snapshot

POLICY = 'V214_AUTHORIZED_2GIB_NORMAL_PRESSURE_GLOBAL_SWAP_TELEMETRY'
LIMITS = dict(max_rss_bytes=2147483648, max_seconds=80,
              min_system_free_percent=50, required_pressure_level=1,
              global_swap_blocking=False)


def snapshot():
    result = legacy_snapshot()
    result['pressure_level'] = int(subprocess.check_output(
        ['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
        text=True, timeout=2).strip())
    return result


def check_sample(sample, previous, rss, elapsed, config):
    if config != LIMITS:
        raise ValueError('Exact authorized V214 memory limits required')
    if (set(sample) != {'swapouts', 'free_percent', 'pressure_level'}
        or any(not isinstance(v, int) or isinstance(v, bool) for v in sample.values())
        or sample['swapouts'] < 0 or not 0 <= sample['free_percent'] <= 100
        or sample['pressure_level'] not in (1, 2, 4)):
        raise ValueError('Missing/invalid actual memory-pressure counters')
    if previous is not None and sample['swapouts'] < previous:
        raise RuntimeError('STOP_GLOBAL_SWAP_COUNTER_RESET')
    if (sample['pressure_level'] != config['required_pressure_level']
        or sample['free_percent'] < config['min_system_free_percent']):
        raise RuntimeError('STOP_ACTUAL_MEMORY_PRESSURE')
    if rss > config['max_rss_bytes'] or elapsed > config['max_seconds']:
        raise RuntimeError('STOP_RSS_OR_WALL_RESOURCE_POLICY')


def quiet(seconds, minimum, maximum=55):
    if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
        or not 0 < seconds <= maximum <= 55 or minimum != LIMITS['min_system_free_percent']):
        raise ValueError('Bounded authorized pressure preflight required')
    began = time.monotonic(); samples = []; previous = None
    while True:
        sample = snapshot(); now = time.monotonic()
        check_sample(sample, previous, 0, 0, LIMITS)
        samples.append(dict(elapsed_seconds=now-began, **sample))
        previous = sample['swapouts']
        if now-began >= seconds:
            return dict(policy_id=POLICY, pass_system_quiet_policy=True,
                        worker_swap_budget_changed=True, quiet_seconds=now-began,
                        final_snapshot=sample, samples=samples,
                        global_swap_delta_since_first_sample=previous-samples[0]['swapouts'],
                        global_swap_is_worker_attributed=False)
        if now-began >= maximum:
            raise RuntimeError('STOP_PRESSURE_PREFLIGHT_NO_WORKER_LAUNCHED')
        time.sleep(min(.5, seconds-(now-began)))


class Budget:
    def __init__(self, config):
        if config != LIMITS: raise ValueError('Exact authorized V214 memory limits required')
        self.c=config; self.stop=threading.Event(); self.failure=None; self.peak=0
        self.minimum=100; self.pressure_levels=set(); self.samples=0

    def observe(self, rss):
        sample=snapshot(); self.peak=max(self.peak,rss)
        self.minimum=min(self.minimum,sample['free_percent'])
        self.pressure_levels.add(sample['pressure_level']); self.samples+=1
        check_sample(sample,self.after,self.peak,time.monotonic()-self.start,self.c)
        self.after=sample['swapouts']

    def __enter__(self):
        self.start=time.monotonic(); initial=snapshot()
        check_sample(initial,None,psutil.Process().memory_info().rss,0,self.c)
        self.before=self.after=initial['swapouts']
        self.observe(psutil.Process().memory_info().rss)
        self.thread=threading.Thread(target=self._watch,daemon=True); self.thread.start()
        return self

    def _watch(self):
        while not self.stop.wait(.5):
            try: self.observe(psutil.Process().memory_info().rss)
            except Exception as exc:
                self.failure=str(exc); _thread.interrupt_main(); return

    def __exit__(self,*_):
        self.stop.set(); self.thread.join(timeout=5)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try: self.observe(int(peak if sys.platform=='darwin' else peak*1024))
        except Exception as exc: self.failure=self.failure or str(exc)
        self.elapsed=time.monotonic()-self.start
        if self.failure: raise RuntimeError(self.failure)

    def report(self):
        return dict(policy_id=POLICY,wall_seconds=self.elapsed,peak_rss_bytes=self.peak,
                    swapout_pages_delta=self.after-self.before,pass_resource_policy=self.failure is None,
                    minimum_system_free_percent=self.minimum,pressure_levels=sorted(self.pressure_levels),
                    sample_count=self.samples,limits=self.c,global_swap_is_worker_attributed=False)


def validate_report(report):
    if (report.get('policy_id') != POLICY or report.get('limits') != LIMITS
        or report.get('pass_resource_policy') is not True
        or report.get('global_swap_is_worker_attributed') is not False
        or report['peak_rss_bytes'] > LIMITS['max_rss_bytes']
        or report['wall_seconds'] > LIMITS['max_seconds']
        or report['swapout_pages_delta'] < 0 or report['sample_count'] < 2
        or report['minimum_system_free_percent'] < LIMITS['min_system_free_percent']
        or report['pressure_levels'] != [1]):
        raise ValueError('Actual authorized pressure/RSS report required')



import json
import math
from pathlib import Path
from src.xgb_matcher.retrieval_v20_workbench import record

def parent_exit(directory):
    """Worker PASS alone is not proof of a supervised successful child exit."""
    path = directory/'supervision.json'
    if not path.is_file():
        raise ValueError('UNCONFIRMED_PARENT_EXIT_NO_COMPLETE_STORE')
    s = json.loads(path.read_text())
    if (s.get('policy_id') != POLICY or s.get('limits') != LIMITS
            or type(s.get('exit_code')) is not int or s['exit_code'] != 0
            or s.get('own_child_exit_confirmed') is not True
            or type(s.get('child_pid')) is not int or s['child_pid'] <= 0
            or type(s.get('peak_rss_bytes')) is not int or not 0 <= s['peak_rss_bytes'] <= LIMITS['max_rss_bytes']
            or type(s.get('minimum_system_free_percent')) is not int
            or not LIMITS['min_system_free_percent'] <= s['minimum_system_free_percent'] <= 100
            or type(s.get('sample_count')) is not int or s['sample_count'] < 1
            or type(s.get('swapout_pages_delta')) is not int or s['swapout_pages_delta'] < 0
            or s.get('global_swap_is_worker_attributed') is not False
            or not isinstance(s.get('wall_seconds'), (float, int)) or isinstance(s['wall_seconds'], bool)
            or not math.isfinite(s['wall_seconds']) or not 0 <= s['wall_seconds'] <= LIMITS['max_seconds']):
        raise ValueError('UNCONFIRMED_PARENT_EXIT_NO_COMPLETE_STORE')
    return record(path)

