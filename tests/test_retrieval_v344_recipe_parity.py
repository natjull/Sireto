"""Only RAM tolerance changes; fixtures do not claim national quality."""
import inspect

from scripts import retrieval_v333_corroborated_population as old
from scripts import retrieval_v344_corroborated_population as new
from src.xgb_matcher import retrieval_memory_policy_v214 as old_policy
from src.xgb_matcher import retrieval_memory_policy_v342 as new_policy


def test_resource_delta_is_only_rss():
    assert new_policy.LIMITS == dict(old_policy.LIMITS, max_rss_bytes=3*2**30)
    assert old_policy.LIMITS['max_rss_bytes'] == 2*2**30
    before, after = old.config(), new.config()
    assert after == dict(before, resources=new_policy.LIMITS,
        output_root=before['output_root'].replace('v333', 'v344'))


def test_native_search_feature_surface_and_serving_identical():
    assert new.search is old.search
    assert new.feature_query is old.feature_query
    assert new.population is old.population
    assert new.training is old.training
    for name in ('serving', 'checked_update', 'valid_rows', 'positions'):
        assert inspect.getsource(getattr(new, name)) == inspect.getsource(getattr(old, name))
    assert inspect.getsource(new.worker) == inspect.getsource(old.worker).replace('V333', 'V344')


def test_measured_failed_peak_is_accepted_without_relaxing_pressure():
    sample = dict(swapouts=42, free_percent=75, pressure_level=1)
    new_policy.check_sample(sample, 42, 2152611840, 19.116343, new_policy.LIMITS)
