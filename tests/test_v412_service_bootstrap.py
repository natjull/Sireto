from __future__ import annotations

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
