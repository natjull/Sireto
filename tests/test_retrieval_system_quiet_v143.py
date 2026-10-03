import pytest

import src.xgb_matcher.retrieval_system_quiet_v143 as gate


def test_swap_burst_and_low_free_reset_quiet_window_not_worker_budget():
    q=gate.QuietWindow(3,50)
    assert not q.observe(0,100,75)
    assert not q.observe(2,100,75)
    assert not q.observe(3,101,75)
    assert not q.observe(5,101,49)
    assert not q.observe(6,101,75)
    assert not q.observe(8,101,75)
    assert q.observe(9,101,75)
    with pytest.raises(ValueError,match='counter reset'):q.observe(10,99,75)
    with pytest.raises(ValueError,match='counters'):q.observe(11,True,75)


def test_bounded_gate_reports_global_burst_before_successful_stable_window(monkeypatch):
    clock=[0];values=iter([100,100,101,101,101])
    monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(gate.time,'sleep',lambda _:None)
    def sample():
        clock[0]+=1;return dict(swapouts=next(values),free_percent=73)
    monkeypatch.setattr(gate,'snapshot',sample);r=gate.quiet(2,50,5)
    assert r['pass_system_quiet_policy'] and r['global_swap_delta_since_first_sample']==1
    assert r['quiet_seconds']==2 and r['global_swap_delta_in_successful_quiet_window']==0
    assert not r['worker_swap_budget_changed'] and r['final_snapshot']['swapouts']==101


def test_never_quiet_stops_instead_of_launch_or_threshold_relaxation(monkeypatch):
    clock=[0];monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(gate.time,'sleep',lambda _:None)
    def sample():
        clock[0]+=1;return dict(swapouts=100,free_percent=49)
    monkeypatch.setattr(gate,'snapshot',sample)
    with pytest.raises(RuntimeError,match='NO_WORKER_LAUNCHED'):gate.quiet(2,50,3)
    with pytest.raises(ValueError,match='bounded'):gate.quiet(2,50,60)


def test_readonly_actual_counter_parser_no_guessed_pressure(monkeypatch):
    monkeypatch.setattr(gate.subprocess,'check_output',lambda *a,**k:'System-wide memory free percentage: 73%\n')
    monkeypatch.setattr(gate.Budget,'swap',staticmethod(lambda:123))
    assert gate.snapshot()==dict(swapouts=123,free_percent=73)
    monkeypatch.setattr(gate.subprocess,'check_output',lambda *a,**k:'not available')
    with pytest.raises(RuntimeError,match='Missing actual'):gate.snapshot()
