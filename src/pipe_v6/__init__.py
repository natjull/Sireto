"""Pipe V6 package: CRM ↔ SIRENE matching pipeline."""

from .config import PipelineConfig, load_config

__all__ = [
    "PipelineConfig",
    "load_config",
]
