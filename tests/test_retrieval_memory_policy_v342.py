"""Resource control fixtures are simulated, not evidence of national recall."""
import json

import pytest

from src.xgb_matcher import retrieval_memory_policy_v342 as policy
from src.xgb_matcher import retrieval_memory_policy_v214 as old
from src.xgb_matcher.retrieval_v20_workbench import write_json
from test_seal_retrieval_v175_address_parts import supervision


def test_only_explicit_ram_change_old_policy_immutable():
    assert policy.LIMITS == dict(old.LIMITS, max_rss_bytes=3*2**30)
    assert old.LIMITS['max_rss_bytes'] == 2**31
    sample = dict(swapouts=42, free_percent=83, pressure_level=1)
    policy.check_sample(sample, 42, 2152611840, 12., policy.LIMITS)
    with pytest.raises(RuntimeError):
        old.check_sample(sample, 42, 2152611840, 12., old.LIMITS)
    with pytest.raises(ValueError):
        policy.Budget(dict(policy.LIMITS, max_seconds=81))


@pytest.mark.parametrize('change', ['rss', 'wall', 'pressure', 'free', 'reset'])
def test_other_resource_protections_remain_eliminatory(change):
    sample = dict(swapouts=42, free_percent=83, pressure_level=1)
    rss, wall = 1, 1.
    if change == 'rss': rss = 3*2**30+1
    elif change == 'wall': wall = 80.001
    elif change == 'pressure': sample['pressure_level'] = 2
    elif change == 'free': sample['free_percent'] = 49
    else: sample['swapouts'] = 41
    with pytest.raises(RuntimeError):
        policy.check_sample(sample, 42, rss, wall, policy.LIMITS)


@pytest.mark.parametrize('change', ['legacy', 'nonzero', 'unconfirmed', 'rss', 'wall'])
def test_actual_parent_receipt_under_new_policy_required(tmp_path, change):
    receipt = supervision()
    receipt.update(policy_id=policy.POLICY, limits=policy.LIMITS, peak_rss_bytes=2152611840)
    write_json(tmp_path/'supervision.json', receipt)
    assert policy.parent_exit(tmp_path)
    if change == 'legacy': receipt['policy_id'] = old.POLICY
    elif change == 'nonzero': receipt['exit_code'] = 1
    elif change == 'unconfirmed': receipt['own_child_exit_confirmed'] = False
    elif change == 'rss': receipt['peak_rss_bytes'] = 3*2**30+1
    else: receipt['wall_seconds'] = 80.1
    write_json(tmp_path/'supervision.json', receipt)
    with pytest.raises(ValueError):
        policy.parent_exit(tmp_path)
