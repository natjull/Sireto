import subprocess

import psutil
import pytest

from src.xgb_matcher import retrieval_supervisor_v157 as policy


class Child:
    pid=12345
    def __init__(self,code=0):self.returncode=None;self.code=code;self.waited=False;self.terminated=False
    def poll(self):return self.returncode
    def wait(self,timeout):self.waited=True;self.returncode=self.code;return self.code
    def terminate(self):self.terminated=True;self.code=-15
    def kill(self):self.terminated=True;self.code=-9


def setup(monkeypatch,code=0):
    child=Child(code);monkeypatch.setattr(policy.subprocess,'Popen',lambda *a,**k:child)
    counter=iter(range(42,1000))
    monkeypatch.setattr(policy,'snapshot',lambda:dict(swapouts=next(counter),free_percent=73,pressure_level=1))
    def missing(pid):raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(policy.psutil,'Process',missing)
    return child


def test_disappearance_only_succeeds_after_owned_child_actual_exit_zero(monkeypatch,tmp_path):
    child=setup(monkeypatch)
    with (tmp_path/'log').open('x') as log:m=policy.supervise(['own-child'],log,{})
    assert child.waited and m['exit_code']==0 and m['own_child_exit_confirmed']
    assert m['child_disappeared_during_sampling'] and not child.terminated


@pytest.mark.parametrize('code',[1,-9,137])
def test_disappeared_child_nonzero_exit_never_claims_success(monkeypatch,tmp_path,code):
    child=setup(monkeypatch,code)
    with (tmp_path/'log').open('x') as log:
        with pytest.raises(subprocess.CalledProcessError):policy.supervise(['own-child'],log,{})
    assert child.waited


def test_final_pressure_still_fails_even_with_exit_zero(monkeypatch,tmp_path):
    child=setup(monkeypatch)
    values=iter([dict(swapouts=42,free_percent=73,pressure_level=1),dict(swapouts=43,free_percent=49,pressure_level=2)])
    monkeypatch.setattr(policy,'snapshot',lambda:next(values))
    with (tmp_path/'log').open('x') as log:
        with pytest.raises(RuntimeError,match='MEMORY_PRESSURE'):policy.supervise(['own-child'],log,{})
    assert child.waited


def test_missing_pid_with_unconfirmed_exit_times_out_not_success(monkeypatch,tmp_path):
    child=setup(monkeypatch)
    def timeout(timeout):raise subprocess.TimeoutExpired('own-child',timeout)
    monkeypatch.setattr(child,'wait',timeout)
    monkeypatch.setattr(child,'kill',lambda:None)
    with (tmp_path/'log').open('x') as log:
        with pytest.raises(subprocess.TimeoutExpired):policy.supervise(['own-child'],log,{})
    assert child.terminated


def test_pressure_observed_before_disappearance_is_not_discarded(monkeypatch,tmp_path):
    child=setup(monkeypatch)
    monkeypatch.setattr(policy,'snapshot',lambda:dict(swapouts=42,free_percent=49,pressure_level=2))
    with (tmp_path/'log').open('x') as log:
        with pytest.raises(RuntimeError,match='MEMORY_PRESSURE'):policy.supervise(['own-child'],log,{})
    assert child.terminated
