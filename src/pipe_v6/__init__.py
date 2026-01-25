"""Pipe V6/V7 package: CRM ↔ SIRENE matching pipeline.

V7 adds Places-guided candidate generation as an alternative to legacy web search.
"""

from .config import PipelineConfig, load_config

# V7 Places modules
from .serper_places_client import (
    search_places,
    build_places_query,
    PlacesResult,
    PlacesSearchResponse,
)
from .places_candidate_generator import (
    SireneCandidate,
    generate_candidates_by_address,
    generate_candidates_by_radius,
    merge_candidate_pools,
)
from .places_validator import (
    validate_places_candidate,
    make_places_decision,
    PlacesDecisionResult,
)
from .places_orchestrator import (
    CrmRow,
    XgbTopkCandidate,
    process_review_case,
    PlacesProcessingResult,
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
    # Candidate generation
    "SireneCandidate",
    "generate_candidates_by_address",
    "generate_candidates_by_radius",
    "merge_candidate_pools",
    # Validation
    "validate_places_candidate",
    "make_places_decision",
    "PlacesDecisionResult",
    # Orchestration
    "CrmRow",
    "XgbTopkCandidate",
    "process_review_case",
    "PlacesProcessingResult",
]
