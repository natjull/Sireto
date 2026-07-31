"""Fail-closed loader for the frozen persistent V4.12 service bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import joblib
import xgboost as xgb

from . import v411_acceptor as _v411_acceptor  # noqa: F401
from .v411_scene import V411_ACCEPTOR_FEATURE_NAMES
from .v49_site_function import SiteFunctionTaxonomy
from .v412_evidence_service import V412DirectEvidenceService
from .v412_service import (
    FIXED_THRESHOLD,
    RANKER_C_FEATURE_ORDER,
    RANKER_C_FEATURE_ORDER_SHA256,
    V412DownstreamService,
)
from .v412_service_retrieval import V412RetrievalFeatureService
from .v412_strict_stores import (
    StrictPartitionStore,
    StrictSnapshotLookup,
    StrictVerifiedTfidfCache,
)
from .v412_unit_retrieval import CACHE_NAMESPACE


STOP = "STOP_V412_SERVICE_INTEGRITY"
CERTIFICATION_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/certifications/"
    "v4_12_strict_stores/"
    "9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4"
)
RANKER_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_ranker_c/"
    "e13eb3ac7498256e"
)
ACCEPTOR_BUNDLE_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_acceptor/"
    "9d23bf3deb6b63de/bundle"
)
TAXONOMY_PATH = Path(
    "/Users/nathanjullia/Documents/Projets/SIRETO/"
    "config/v4_9_site_function_taxonomy.json"
)

EXPECTED_FILES = {
    "certification_manifest": (
        CERTIFICATION_ROOT / "manifest.json",
        "6854b115f06eb1b1d0dacc8dfc44e2d7d6d470427d80628253ca09a79a443d2d",
    ),
    "strict_run_spec": (
        CERTIFICATION_ROOT / "run_spec.json",
        "c53d08dafb0d0c32a60de51737199191b5aad5d2b39a1649a4a76d58508138bc",
    ),
    "lookup_descriptor": (
        CERTIFICATION_ROOT / "lookup_descriptor.json",
        "3e58872c94b17b5e19dee762e3c33f2bcf7407f948c902aa398b74705d34ce6c",
    ),
    "ranker_manifest": (
        RANKER_ROOT / "manifest.json",
        "1552ab2623580f1ae68e31ec1497be8a93a1bb1f2d33114dd34cfea07a864053",
    ),
    "ranker_metadata": (
        RANKER_ROOT / "ranker_c/metadata.json",
        "120441701dd91865eefac886cf0fab646829b681ffcf2f367d1a28766588ff67",
    ),
    "ranker_model": (
        RANKER_ROOT / "ranker_c/full_fit.json",
        "f4b71b49ed4f879b88e05e4fb84229d0306c5e8ca96958ac20ad97fcc04349c0",
    ),
    "stack_manifest": (
        ACCEPTOR_BUNDLE_ROOT / "stack_manifest.json",
        "81279978f47e1e2b1b4a1ea85d595b8dedd8ee8a073e34a19b3ffd340c945d5a",
    ),
    "acceptor_metadata": (
        ACCEPTOR_BUNDLE_ROOT / "metadata.json",
        "e4b99676e695d19748b71a7657ff5a1f5c7dfa2879754dd2e1b15c8906a61d6b",
    ),
    "acceptor_model": (
        ACCEPTOR_BUNDLE_ROOT / "acceptor_model.joblib",
        "a804feb64f28c417adda4418724f53df50b20d3d308b3e7c778c7189d368e3cf",
    ),
    "taxonomy": (
        TAXONOMY_PATH,
        "48bbb7e1795a0731f1f12df41aeb971667c10d03c879bf06d5ba15b65f8b121d",
    ),
}


def _fail(detail: str) -> None:
    raise ValueError(f"{STOP}: {detail}")


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _capture_exact(path: Path, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        _fail(f"cannot open frozen file {path}: errno={exc.errno}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"frozen path is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        if _identity(after) != _identity(before) or len(payload) != before.st_size:
            _fail(f"frozen file changed while read: {path}")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            _fail(f"frozen file hash changed: {path}")
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            _fail(f"frozen path disappeared: {path}: errno={exc.errno}")
        if _identity(current) != _identity(before):
            _fail(f"frozen path identity changed: {path}")
        return payload
    finally:
        os.close(descriptor)


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail(f"duplicate JSON key in {label}")
            result[key] = value
        return result

    try:
        parsed = json.loads(payload, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON in {label}: {exc}")
    if type(parsed) is not dict:
        _fail(f"{label} must be a JSON object")
    return parsed


def _capture_bundle_files() -> dict[str, bytes]:
    return {
        role: _capture_exact(path, digest)
        for role, (path, digest) in EXPECTED_FILES.items()
    }


def _validate_control_files(
    payloads: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    certification = _json_object(
        payloads["certification_manifest"],
        "certification manifest",
    )
    if (
        certification.get("schema_version")
        != "sireto-v4.12-strict-stores-certification-1"
        or certification.get("build_id")
        != "9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4"
        or certification.get("verdict") != "GO_V412_STRICT_STORES_SANDBOX"
    ):
        _fail("strict-store certification changed")
    certified_files = {
        item.get("path"): item
        for item in certification.get("files", [])
        if type(item) is dict
    }
    for name, role in (
        ("run_spec.json", "strict_run_spec"),
        ("lookup_descriptor.json", "lookup_descriptor"),
    ):
        record = certified_files.get(name)
        if (
            type(record) is not dict
            or record.get("sha256")
            != hashlib.sha256(payloads[role]).hexdigest()
            or record.get("size_bytes") != len(payloads[role])
        ):
            _fail(f"certification does not bind {name}")

    run_spec = _json_object(payloads["strict_run_spec"], "strict run spec")
    descriptor = _json_object(
        payloads["lookup_descriptor"],
        "lookup descriptor",
    )
    if (
        run_spec.get("schema_version")
        != "sireto-v4.12-strict-stores-run-spec-1"
        or run_spec.get("query_count") != 1456
        or run_spec.get("lookup_descriptor_sha256")
        != hashlib.sha256(payloads["lookup_descriptor"]).hexdigest()
        or run_spec.get("max_rss_bytes") != 8 * 1024**3
    ):
        _fail("strict run specification changed")
    return certification, run_spec, descriptor


def _validate_model_controls(payloads: Mapping[str, bytes]) -> None:
    ranker_manifest = _json_object(
        payloads["ranker_manifest"],
        "ranker manifest",
    )
    ranker_metadata = _json_object(
        payloads["ranker_metadata"],
        "ranker metadata",
    )
    for document in (ranker_manifest, ranker_metadata):
        if (
            document.get("build_id") != "e13eb3ac7498256e"
            or tuple(document.get("feature_order") or ())
            != RANKER_C_FEATURE_ORDER
            or document.get("feature_order_sha256")
            != RANKER_C_FEATURE_ORDER_SHA256
        ):
            _fail("Ranker C metadata changed")
    outputs = ranker_manifest.get("outputs")
    full_fit = (
        outputs.get("ranker_c/full_fit.json")
        if type(outputs) is dict
        else None
    )
    if (
        type(full_fit) is not dict
        or full_fit.get("sha256")
        != hashlib.sha256(payloads["ranker_model"]).hexdigest()
        or full_fit.get("size_bytes") != len(payloads["ranker_model"])
    ):
        _fail("Ranker C manifest/model mismatch")

    stack = _json_object(payloads["stack_manifest"], "stack manifest")
    acceptor_metadata = _json_object(
        payloads["acceptor_metadata"],
        "acceptor metadata",
    )
    if (
        stack.get("schema_version") != "sireto-v4.11-end-to-end-bundle-1"
        or stack.get("model_bundle_id") != "9d23bf3deb6b63de"
        or stack.get("decision_rule")
        != "AUTO_MATCH if score >= threshold else REVIEW"
        or stack.get("unresolved_policy") != "FORCE_REVIEW"
    ):
        _fail("stack manifest changed")
    components = stack.get("components")
    if type(components) is not dict:
        _fail("stack components missing")
    expected_component_hashes = {
        ("ranker_c", "manifest_sha256"): hashlib.sha256(
            payloads["ranker_manifest"]
        ).hexdigest(),
        ("ranker_c", "model_sha256"): hashlib.sha256(
            payloads["ranker_model"]
        ).hexdigest(),
        ("acceptor", "metadata_sha256"): hashlib.sha256(
            payloads["acceptor_metadata"]
        ).hexdigest(),
        ("acceptor", "model_sha256"): hashlib.sha256(
            payloads["acceptor_model"]
        ).hexdigest(),
        ("scene", "taxonomy_sha256"): hashlib.sha256(
            payloads["taxonomy"]
        ).hexdigest(),
    }
    for (component, field), expected in expected_component_hashes.items():
        value = components.get(component)
        if type(value) is not dict or value.get(field) != expected:
            _fail(f"stack component changed: {component}.{field}")
    if (
        acceptor_metadata.get("schema_version")
        != "sireto-v4.11-acceptor-bundle-1"
        or acceptor_metadata.get("model_bundle_id") != "9d23bf3deb6b63de"
        or acceptor_metadata.get("model_family") != "COMPACT_LOGIT"
        or acceptor_metadata.get("threshold") != FIXED_THRESHOLD
        or acceptor_metadata.get("decision_rule")
        != "AUTO_MATCH if score >= threshold else REVIEW"
        or acceptor_metadata.get("unresolved_policy") != "FORCE_REVIEW"
        or acceptor_metadata.get("feature_order")
        != V411_ACCEPTOR_FEATURE_NAMES
    ):
        _fail("acceptor metadata changed")


@dataclass
class FrozenV412ServiceBundle:
    partition_store: StrictPartitionStore
    tfidf_cache: StrictVerifiedTfidfCache
    lookup: StrictSnapshotLookup
    retrieval: V412RetrievalFeatureService
    downstream: V412DownstreamService
    evidence: V412DirectEvidenceService | None
    asset_hashes: dict[str, str]
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            self.lookup.close()
            self._closed = True

    def __enter__(self) -> "FrozenV412ServiceBundle":
        if self._closed:
            _fail("cannot reopen a closed service bundle")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def load_frozen_v412_service_bundle(
    *,
    include_evidence: bool,
) -> FrozenV412ServiceBundle:
    """Load the one frozen production stack; no paths or models are injectable."""
    if type(include_evidence) is not bool:
        _fail("include_evidence must be boolean")
    payloads = _capture_bundle_files()
    _, run_spec, descriptor = _validate_control_files(payloads)
    _validate_model_controls(payloads)
    allowed = run_spec.get("allowed_read_files")
    partition_records = run_spec.get("partition_records")
    cache_records = run_spec.get("cache_records")
    if not all(
        isinstance(value, list)
        for value in (allowed, partition_records, cache_records)
    ):
        _fail("strict store records missing")

    partition_store = StrictPartitionStore(
        partition_records,
        allowed,
        max_cache_entries=5,
    )
    tfidf_cache = StrictVerifiedTfidfCache(
        cache_records,
        allowed,
        namespace=CACHE_NAMESPACE,
        max_cache_entries=20,
    )
    if partition_store.partition_keys != tfidf_cache.partition_keys:
        _fail("strict partition and TF-IDF keysets differ")
    lookup = StrictSnapshotLookup(descriptor, allowed)
    try:
        ranker = xgb.XGBRanker()
        ranker.load_model(bytearray(payloads["ranker_model"]))
        acceptor = joblib.load(BytesIO(payloads["acceptor_model"]))
        if not callable(getattr(acceptor, "predict_proba", None)):
            _fail("frozen acceptor has no predict_proba")
        taxonomy = SiteFunctionTaxonomy(
            _json_object(payloads["taxonomy"], "taxonomy")
        )
        retrieval = V412RetrievalFeatureService(
            partition_store=partition_store,
            tfidf_cache=tfidf_cache,
            lookup=lookup,
            ranker_feature_order=RANKER_C_FEATURE_ORDER,
        )
        downstream = V412DownstreamService(
            ranker=ranker,
            acceptor=acceptor,
            taxonomy=taxonomy,
            ranker_feature_order=RANKER_C_FEATURE_ORDER,
        )
        evidence = (
            V412DirectEvidenceService(partition_store=partition_store)
            if include_evidence
            else None
        )
        return FrozenV412ServiceBundle(
            partition_store=partition_store,
            tfidf_cache=tfidf_cache,
            lookup=lookup,
            retrieval=retrieval,
            downstream=downstream,
            evidence=evidence,
            asset_hashes={
                role: hashlib.sha256(payload).hexdigest()
                for role, payload in payloads.items()
            },
        )
    except BaseException:
        lookup.close()
        raise


__all__ = [
    "FrozenV412ServiceBundle",
    "load_frozen_v412_service_bundle",
]
