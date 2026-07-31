from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_v412_persistent_service_bootstrap import _audit


@pytest.mark.parametrize(
    "event",
    (
        "socket.connect",
        "socket.getaddrinfo",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.fork",
    ),
)
def test_bootstrap_denies_network_and_child_process_events(event: str) -> None:
    with pytest.raises(PermissionError, match="syscall denied"):
        _audit(event, ())


def test_frozen_bundle_loads_after_real_bootstrap_audit_hook() -> None:
    repository = Path(__file__).resolve().parent.parent
    code = f"""
import sys
sys.path.insert(0, {str(repository)!r})
from scripts.run_v412_persistent_service_bootstrap import _audit
sys.addaudithook(_audit)
from src.xgb_matcher.v412_service_bundle import (
    load_frozen_v412_service_bundle,
)
with load_frozen_v412_service_bundle(include_evidence=False):
    pass
print("BUNDLE_LOADED_UNDER_AUDIT_HOOK")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BUNDLE_LOADED_UNDER_AUDIT_HOOK"
