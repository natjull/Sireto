"""Logging helpers for Pipe V6."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import PipelineConfig


def _resolve_level(level_name: str) -> int:
    try:
        return getattr(logging, level_name.upper())
    except AttributeError:
        return logging.INFO


def setup_logging(
    config: PipelineConfig,
    *,
    logger_name: str = "pipe_v6",
    stream: Optional[logging.Handler] = None,
) -> logging.Logger:
    """Configure and return a logger according to the pipeline configuration."""

    logger = logging.getLogger(logger_name)
    if getattr(logger, "_pipe_v6_configured", False):
        return logger

    log_level = _resolve_level(config.log_level)
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path: Path = config.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = stream or logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    setattr(logger, "_pipe_v6_configured", True)
    return logger
