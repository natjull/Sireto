"""Versioned V4.1 release manifest with explicit component compatibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .v41_acceptor import V41_CONFIDENCE_KIND
from .v41_features import validate_v41_model_feature_order


@dataclass(frozen=True)
class V41ReleaseManifest:
    release_id: str
    retrieval_signature: str
    ranker_bundle_id: str
    acceptor_bundle_id: str
    ranker_dataset_manifest_id: str
    acceptor_dataset_manifest_id: str
    ranker_feature_order: list[str]
    acceptor_feature_order: list[str]
    ranker_variant: str
    schema_version: str = "v4.1-release-1"
    confidence_kind: str = V41_CONFIDENCE_KIND

    @classmethod
    def build(
        cls,
        *,
        retrieval_signature: str,
        ranker_bundle_id: str,
        acceptor_bundle_id: str,
        ranker_dataset_manifest_id: str,
        acceptor_dataset_manifest_id: str,
        ranker_feature_order: list[str],
        acceptor_feature_order: list[str],
        ranker_variant: str,
    ) -> "V41ReleaseManifest":
        if ranker_variant not in {"R0", "R1"}:
            raise ValueError("ranker_variant must be R0 or R1")
        validate_v41_model_feature_order(
            ranker_feature_order,
            require_v41_features=ranker_variant == "R1",
        )
        identity = {
            "retrieval_signature": retrieval_signature,
            "ranker_bundle_id": ranker_bundle_id,
            "acceptor_bundle_id": acceptor_bundle_id,
            "ranker_dataset_manifest_id": ranker_dataset_manifest_id,
            "acceptor_dataset_manifest_id": acceptor_dataset_manifest_id,
            "ranker_feature_order": ranker_feature_order,
            "acceptor_feature_order": acceptor_feature_order,
            "ranker_variant": ranker_variant,
            "confidence_kind": V41_CONFIDENCE_KIND,
        }
        release_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return cls(release_id=release_id, **identity)

    def validate_components(
        self,
        *,
        ranker_metadata: Mapping[str, Any],
        acceptor_metadata: Mapping[str, Any],
    ) -> None:
        """Validate each component independently; dataset IDs need not match."""
        checks = {
            "ranker bundle": (
                ranker_metadata.get("model_bundle_id"),
                self.ranker_bundle_id,
            ),
            "acceptor bundle": (
                acceptor_metadata.get("model_bundle_id"),
                self.acceptor_bundle_id,
            ),
            "ranker dataset": (
                ranker_metadata.get("dataset_manifest_id"),
                self.ranker_dataset_manifest_id,
            ),
            "acceptor dataset": (
                acceptor_metadata.get("dataset_manifest_id"),
                self.acceptor_dataset_manifest_id,
            ),
            "ranker retrieval signature": (
                ranker_metadata.get("retrieval_signature"),
                self.retrieval_signature,
            ),
            "acceptor retrieval signature": (
                acceptor_metadata.get("retrieval_signature"),
                self.retrieval_signature,
            ),
            "ranker feature order": (
                list(ranker_metadata.get("feature_order") or []),
                self.ranker_feature_order,
            ),
            "ranker variant": (
                ranker_metadata.get("ranker_variant"),
                self.ranker_variant,
            ),
            "acceptor feature order": (
                list(acceptor_metadata.get("feature_order") or []),
                self.acceptor_feature_order,
            ),
            "acceptor calibration": (
                acceptor_metadata.get("calibration_method"),
                "raw",
            ),
            "acceptor confidence kind": (
                acceptor_metadata.get("confidence_kind"),
                V41_CONFIDENCE_KIND,
            ),
        }
        mismatches = [
            name for name, (actual, expected) in checks.items() if actual != expected
        ]
        if mismatches:
            raise ValueError(f"Incompatible V4.1 release components: {mismatches}")

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "V41ReleaseManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(**payload)
        if manifest.schema_version != "v4.1-release-1":
            raise ValueError("Unsupported V4.1 release manifest")
        return manifest


__all__ = ["V41ReleaseManifest"]
