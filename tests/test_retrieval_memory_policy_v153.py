import pytest

from src.xgb_matcher import retrieval_memory_policy_v153 as policy


def sample(swap=42, free=73, pressure=1):
    return dict(swapouts=swap, free_percent=free, pressure_level=pressure)


def test_global_swap_increase_is_telemetry_not_attributed_worker_failure():
    policy.check_sample(sample(4200000),42,781516800,12.7,policy.LIMITS)
    with pytest.raises(RuntimeError,match='COUNTER_RESET'):
        policy.check_sample(sample(41),42,0,0,policy.LIMITS)


@pytest.mark.parametrize('s,rss,elapsed,message',[
    (sample(pressure=2),0,0,'PRESSURE'),(sample(pressure=4),0,0,'PRESSURE'),
    (sample(free=49),0,0,'PRESSURE'),
    (sample(),1073741825,0,'RSS_OR_WALL'),(sample(),0,80.001,'RSS_OR_WALL'),
])
def test_actual_resource_failure_stops(s,rss,elapsed,message):
    with pytest.raises(RuntimeError,match=message):
        policy.check_sample(s,42,rss,elapsed,policy.LIMITS)


@pytest.mark.parametrize('s',[{},sample(-1),sample(free=101),sample(pressure=0),sample(swap=True)])
def test_missing_or_invalid_counters_never_inferred_normal(s):
    with pytest.raises(ValueError,match='counters'):
        policy.check_sample(s,None,0,0,policy.LIMITS)


def test_no_silent_relaxation():
    with pytest.raises(ValueError,match='Exact authorized'):
        policy.Budget(dict(policy.LIMITS,max_rss_bytes=3221225472))


def test_quiet_accepts_global_increase_and_records_new_policy(monkeypatch):
    times=iter([0,0,1]); swaps=iter([42,45])
    monkeypatch.setattr(policy.time,'monotonic',lambda:next(times))
    monkeypatch.setattr(policy.time,'sleep',lambda _:None)
    monkeypatch.setattr(policy,'snapshot',lambda:sample(next(swaps)))
    gate=policy.quiet(1,50)
    assert gate['policy_id']==policy.POLICY and gate['worker_swap_budget_changed']
    assert gate['global_swap_delta_since_first_sample']==3
    assert gate['global_swap_is_worker_attributed'] is False


def test_budget_actual_nonzero_global_swap_is_published(monkeypatch):
    swaps=iter(range(42,1000))
    monkeypatch.setattr(policy,'snapshot',lambda:sample(next(swaps)))
    with policy.Budget(policy.LIMITS) as budget: pass
    report=budget.report();policy.validate_report(report)
    assert report['swapout_pages_delta']>0 and report['pressure_levels']==[1]


def test_supervisor_terminates_only_owned_child_on_real_pressure(monkeypatch,tmp_path):
    class Child:
        pid=987654; returncode=None; terminated=False
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def wait(self,timeout):return self.returncode
    child=Child();monkeypatch.setattr(policy.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(policy,'snapshot',lambda:sample(pressure=2))
    monkeypatch.setattr(policy.psutil,'Process',lambda pid:type('Process',(),{'memory_info':lambda self:type('RSS',(),{'rss':50})()})())
    with (tmp_path/'log').open('x') as log:
        with pytest.raises(RuntimeError,match='PRESSURE'):policy.supervise(['owned-child'],log,{})
    assert child.terminated
