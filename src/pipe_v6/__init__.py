"""Pipe V6/V7 package: CRM ↔ SIRENE matching pipeline.

V7 uses simplified Places fallback: "Places as CRM Repair".
"""

from .config import PipelineConfig, load_config

# V7 Places modules
from .serper_places_client import (
    search_places,
    build_places_query,
    PlacesResult,
    PlacesSearchResponse,
)
from .places_fallback import (
    fallback_with_places,
    PlacesFallbackResult,
)

__all__ = [
    # Config
    "PipelineConfig",
    "load_config",
    # Places API
    "search_places",
    "build_places_query",
    "PlacesResult",
    "PlacesSearchResponse",
    # Places Fallback (V7 simplified)
    "fallback_with_places",
    "PlacesFallbackResult",
]
