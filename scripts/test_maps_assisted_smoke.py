from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parent
builder_spec = importlib.util.spec_from_file_location("build_synthetic_gt_corpus", ROOT / "build_synthetic_gt_corpus.py")
assert builder_spec and builder_spec.loader
builder = importlib.util.module_from_spec(builder_spec)
import sys

sys.modules["build_synthetic_gt_corpus"] = builder
builder_spec.loader.exec_module(builder)

maps_spec = importlib.util.spec_from_file_location("maps_assisted_smoke", ROOT / "maps_assisted_smoke.py")
assert maps_spec and maps_spec.loader
maps = importlib.util.module_from_spec(maps_spec)
maps_spec.loader.exec_module(maps)


def test_environment_secret_is_never_returned_or_logged() -> None:
    fake = "fake-secret-for-redaction-test"
    plan = {"maps_assisted": {"enabled": True, "max_requests": 100, "daily_quota": 100, "max_cost_eur": 1.0, "cost_per_request_eur": 0.01, "field_mask": maps.FIELD_MASK, "query_version": maps.QUERY_VERSION}}
    result = maps.maps_preflight(plan, environ={maps.SECRET_ENV: fake})
    serialized = repr(result)
    assert result["status"] == "READY"
    assert fake not in serialized
    smoke = maps.run_smoke(plan, [], execute=False, environ={maps.SECRET_ENV: fake})
    assert fake not in repr(smoke)


def test_keychain_fallback_uses_exact_non_shell_command_without_exposing_stdout() -> None:
    fake = "keychain-fake-secret"
    calls: list[dict[str, object]] = []

    def runner(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return SimpleNamespace(returncode=0, stdout=fake + "\n", stderr="secret-like-error")

    value, source = maps.read_secret(environ={}, keychain_runner=runner)
    assert value == fake
    assert source == "KEYCHAIN"
    assert calls[0]["command"][:6] == ["security", "find-generic-password", "-a", maps.getpass.getuser(), "-s", maps.KEYCHAIN_SERVICE]
    assert calls[0]["command"][-1] == "-w"
    assert calls[0]["shell"] is False
    assert calls[0]["capture_output"] is True
    assert calls[0]["text"] is True


def test_missing_secret_blocks_without_calling_request() -> None:
    plan = {"maps_assisted": {"enabled": True, "max_requests": 100, "daily_quota": 100, "max_cost_eur": 1.0, "cost_per_request_eur": 0.01}}
    result = maps.run_smoke(plan, [{"name": "Example", "address": "1 Rue Alpha", "postcode": "75001", "city": "Paris"}], execute=True, environ={}, keychain_runner=lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""))
    assert result["status"] == "NOT_CONFIGURED"
    assert result["calls_attempted"] == 0
