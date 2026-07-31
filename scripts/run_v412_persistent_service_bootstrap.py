#!/usr/bin/env python3
"""Install fail-closed process guards before importing the service worker."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


STOP = "STOP_V412_SERVICE_INTEGRITY"
WORKER = Path(
    "/Users/nathanjullia/Documents/Projets/SIRETO/"
    "scripts/run_v412_persistent_service_worker.py"
)


def _audit(event: str, args: tuple[object, ...]) -> None:
    if (
        event.startswith("socket.")
        or event
        in {
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.posix_spawnp",
            "os.fork",
            "pty.spawn",
        }
    ):
        raise PermissionError(f"{STOP}: process/network syscall denied")


def main() -> int:
    if os.environ.get("SIRETO_NETWORK_AUDIT_DENY") is not None:
        raise ValueError(f"{STOP}: bootstrap marker pre-seeded")
    sys.addaudithook(_audit)
    os.environ["SIRETO_NETWORK_AUDIT_DENY"] = "1"
    if not WORKER.is_file() or WORKER.is_symlink():
        raise ValueError(f"{STOP}: worker path changed")
    runpy.run_path(str(WORKER), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
