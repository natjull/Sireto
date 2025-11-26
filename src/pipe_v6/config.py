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


def _as_bool(value: Any) -> bool:
    """Best-effort coercion of user-provided values to ``bool``."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PipelineConfig:
    """Centralized configuration values for the Pipe V6 orchestrator."""

    crm_path: Path = _expand_path("data/crm.csv")
    output_path: Path = _expand_path("output/results.csv")
    sqlite_path: Path = _expand_path("data/sirene_cache.sqlite")
    log_path: Path = _expand_path("logs/pipe_v6.log")

    sirene_api_url: str = "https://api.insee.fr/api-sirene/3.11"
    sirene_token: str = ""
    # RNE (INPI) authentication now uses /api/sso/login with username/password.
    # Legacy client_id/client_secret fields are kept for backward compatibility
    # and as a fallback if username/password are not provided.
    rne_api_url: str = "https://registre-national-entreprises.inpi.fr/api/"
    rne_login_url: str = "https://registre-national-entreprises.inpi.fr/api/sso/login"
    rne_username: str = ""
    rne_password: str = ""
    rne_client_id: str = ""  # deprecated
    rne_client_secret: str = ""  # deprecated
    rne_token_url: str = "https://api.inpi.fr/api/sso/oauth2/token"  # deprecated
    # Qwant public API endpoint (v3 as of 2025). The older
    # https://api.qwant.com/api/search/web now returns 404.
    qwant_base_url: str = "https://api.qwant.com/v3/search/web"
    qwant_enabled: bool = True
    datagouv_api_url: str = "https://recherche-entreprises.api.gouv.fr"

    model_name: str = "gpt-oss:20b"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    max_tokens_matcher: int = 2048
    max_tokens_normalizer: int = 1024

    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_sec: float = 160.0
    llm_connect_timeout_sec: float = 5.0
    llm_max_retries: int = 3
    llm_retry_backoff_base_sec: float = 1.0
    llm_max_concurrency: int = 1
    llm_log_prompts: bool = False
    llm_log_responses: bool = False
    llm_json_mode_default: bool = False
    llm_seed: int | None = 42

    max_candidates_per_source: int = 10
    confidence_auto_match: float = 0.85
    confidence_review_min: float = 0.6
    # Candidate filtering (task 7.2)
    category_filter_enabled: bool = True
    category_filter_fallback: bool = True
    max_candidates_llm_matcher: int = 15

    log_level: str = "INFO"
    store_source_raw: bool = True

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
        "sirene_api_url": "https://api.insee.fr/api-sirene/3.11",
        "sirene_token": "",
        # RNE endpoints (June 2025 doc): login + companies root
        "rne_api_url": "https://registre-national-entreprises.inpi.fr/api/",
        "rne_login_url": "https://registre-national-entreprises.inpi.fr/api/sso/login",
        "rne_username": "",
        "rne_password": "",
        # Legacy values for backward compatibility (not used if username/password provided)
        "rne_token_url": "https://api.inpi.fr/api/sso/oauth2/token",
        "rne_client_id": "",
        "rne_client_secret": "",
        # Keep the v3 endpoint as default; the legacy /api/search/web returns 404.
        "qwant_base_url": "https://api.qwant.com/v3/search/web",
        "qwant_enabled": True,
        "datagouv_api_url": "https://recherche-entreprises.api.gouv.fr",
        "model_name": "gpt-oss:20b",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "max_tokens_matcher": 2048,
        "max_tokens_normalizer": 1024,
        "ollama_base_url": "http://localhost:11434",
        "llm_timeout_sec": 160.0,
        "llm_connect_timeout_sec": 5.0,
        "llm_max_retries": 3,
        "llm_retry_backoff_base_sec": 1.0,
        "llm_max_concurrency": 1,
        "llm_log_prompts": False,
        "llm_log_responses": False,
        "llm_json_mode_default": False,
        "llm_seed": 42,
        "max_candidates_per_source": 10,
        "confidence_auto_match": 0.85,
        "confidence_review_min": 0.6,
        "category_filter_enabled": True,
        "category_filter_fallback": True,
        "max_candidates_llm_matcher": 15,
        "log_level": "INFO",
        "store_source_raw": True,
    }

    extracted: dict[str, Any] = {
        key: _from_sources(key, default) for key, default in defaults.items()
    }

    path_keys = {"crm_path", "output_path", "sqlite_path", "log_path"}
    for key in path_keys:
        extracted[key] = _expand_path(extracted[key])

    numeric_casts: dict[str, Any] = {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "max_tokens_matcher": int,
        "max_tokens_normalizer": int,
        "llm_timeout_sec": float,
        "llm_connect_timeout_sec": float,
        "llm_max_retries": int,
        "llm_retry_backoff_base_sec": float,
        "llm_max_concurrency": int,
        "max_candidates_per_source": int,
        "confidence_auto_match": float,
        "confidence_review_min": float,
        "max_candidates_llm_matcher": int,
    }

    for field_name, caster in numeric_casts.items():
        value = extracted[field_name]
        if isinstance(value, str):
            extracted[field_name] = caster(value)

    bool_fields = [
        "store_source_raw",
        "llm_log_prompts",
        "llm_log_responses",
        "llm_json_mode_default",
        "category_filter_enabled",
        "category_filter_fallback",
        "qwant_enabled",
    ]
    for field_name in bool_fields:
        extracted[field_name] = _as_bool(extracted[field_name])

    seed_value = extracted.get("llm_seed")
    if seed_value in ("", None):
        extracted["llm_seed"] = None
    elif isinstance(seed_value, str):
        extracted["llm_seed"] = int(seed_value)
    elif isinstance(seed_value, (int, float)):
        extracted["llm_seed"] = int(seed_value)

    log_level = str(extracted["log_level"]).upper()
    extracted["log_level"] = log_level

    known_keys = set(defaults)
    extras = {
        key: value for key, value in config_data.items() if key not in known_keys
    }

    return PipelineConfig(**extracted, extra=extras)
