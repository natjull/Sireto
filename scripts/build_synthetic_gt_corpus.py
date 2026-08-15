#!/usr/bin/env python3
"""Compatibility entry point for the agentic-only SIRETO GT workflow.

The former mechanical corruption builder was intentionally removed. The only
authorised runtime is the durable lease-based orchestrator below.
"""

from __future__ import annotations

from run_synthetic_gt_agentic_loop import main


if __name__ == "__main__":
    main()
