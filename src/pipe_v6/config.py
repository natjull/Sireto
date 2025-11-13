"""Configuration helpers for the Pipe V6 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping
import os

try:  # Optional dependency: PyYAML is convenient but not mandatory to import.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only when PyYAML missing.
    yaml = None  # type: ignore


def _expand_path(value: str | Path) -> Path:
    """Return an absolute `Path` with user home expanded."""

    return Path(value).expanduser().resolve()


@dataclass
class PipelineConfig:
    """Centralized configuration values for the Pipe V6 orchestrator."""

    crm_path: Path
    output_path: Path
    sqlite_path: Path
    log_path: Path

    sirene_api_url: str
    sirene_token: str
    rne_api_url: str
    rne_client_id: str
    rne_client_secret: str
    qwant_base_url: str

    model_name: str = "gpt-oss:20b"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512

    max_candidates_per_source: int = 10
    confidence_auto_match: float = 0.85
    confidence_review_min: float = 0.6

    log_level: str = "INFO"

    extra: MutableMapping[str, Any] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load configuration from YAML if available, otherwise from environment variables."""

    config_data: Mapping[str, Any] = {}
    config_path: Path | None = None

    if path is not None:
        config_path = _expand_path(path)
    else:
        default = Path("config.yaml")
        if default.exists():
            config_path = default.resolve()

    if config_path and config_path.exists():
        if yaml is None:  # pragma: no cover - triggered when PyYAML missing.
            raise RuntimeError(
                "PyYAML is required to load configuration from YAML files."
            )
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        config_data = loaded or {}

    env_prefix = "SIRETO_"

    def _from_sources(key: str, default: Any) -> Any:
        if key in config_data:
            return config_data[key]
        env_key = env_prefix + key.upper()
        value = os.getenv(env_key)
        return value if value is not None else default

    defaults = {
        "crm_path": "data/crm.csv",
        "output_path": "output/results.csv",
        "sqlite_path": "data/sirene_cache.sqlite",
        "log_path": "logs/pipe_v6.log",
        "sirene_api_url": "https://api.insee.fr/entreprises/sirene/V3",
        "sirene_token": "",
        "rne_api_url": "https://api.inpi.fr/api/rne/",
        "rne_client_id": "",
        "rne_client_secret": "",
        "qwant_base_url": "https://api.qwant.com/api/search/web",
        "model_name": "gpt-oss:20b",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "max_candidates_per_source": 10,
        "confidence_auto_match": 0.85,
        "confidence_review_min": 0.6,
        "log_level": "INFO",
    }

    extracted: dict[str, Any] = {
        key: _from_sources(key, default) for key, default in defaults.items()
    }

    path_keys = {"crm_path", "output_path", "sqlite_path", "log_path"}
    for key in path_keys:
        extracted[key] = _expand_path(extracted[key])

    numeric_fields = {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "max_candidates_per_source": int,
        "confidence_auto_match": float,
        "confidence_review_min": float,
    }

    for field_name, caster in numeric_fields.items():
        value = extracted[field_name]
        if isinstance(value, str):
            extracted[field_name] = caster(value)

    log_level = str(extracted["log_level"]).upper()
    extracted["log_level"] = log_level

    known_keys = set(defaults)
    extras = {
        key: value for key, value in config_data.items() if key not in known_keys
    }

    return PipelineConfig(**extracted, extra=extras)
