"""Budget-preserving fusion for multichannel V9 retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class FusedHit:
    key: Hashable
    rank: int
    rrf_score: float
    channel_ranks: dict[str, int]

    @property
    def source(self) -> str:
        return "+".join(sorted(self.channel_ranks))


def reciprocal_rank_fusion(
    channels: Mapping[str, Sequence[Hashable]],
    *,
    budget: int,
    rrf_k: int = 60,
    channel_weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse ranked channels and truncate to an exact maximum budget."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    weights = dict(channel_weights or {})
    scores: dict[Hashable, float] = {}
    ranks: dict[Hashable, dict[str, int]] = {}

    for channel_name, ordered_keys in channels.items():
        weight = float(weights.get(channel_name, 1.0))
        seen: set[Hashable] = set()
        for rank, key in enumerate(ordered_keys, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            ranks.setdefault(key, {})[channel_name] = rank

    ordered = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            min(ranks[key].values()),
            str(key),
        ),
    )[:budget]
    return [
        FusedHit(
            key=key,
            rank=rank,
            rrf_score=scores[key],
            channel_ranks=ranks[key],
        )
        for rank, key in enumerate(ordered, start=1)
    ]


def annotate_fused_candidate(candidate: dict, hit: FusedHit) -> dict:
    """Return a candidate copy carrying stable retrieval provenance fields."""
    out = dict(candidate)
    out["rrf_score"] = float(hit.rrf_score)
    out["retrieval_rank"] = int(hit.rank)
    out["retrieval_source"] = hit.source
    sparse_ranks = [
        rank
        for channel, rank in hit.channel_ranks.items()
        if channel == "sparse" or channel.startswith("sparse_")
    ]
    out["sparse_rank"] = min(sparse_ranks) if sparse_ranks else None
    out["sparse_name_rank"] = hit.channel_ranks.get("sparse_name")
    out["sparse_address_rank"] = hit.channel_ranks.get("sparse_address")
    out["dense_rank"] = hit.channel_ranks.get("dense")
    out["global_siren_rank"] = hit.channel_ranks.get("global_siren")
    out["rescue_rank"] = hit.channel_ranks.get("rescue")
    out["retrieval_channel_count"] = len(hit.channel_ranks)
    out["retrieval_agreement"] = int(
        bool(sparse_ranks) and "dense" in hit.channel_ranks
    )
    return out


__all__ = ["FusedHit", "reciprocal_rank_fusion", "annotate_fused_candidate"]
