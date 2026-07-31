"""Fail-closed loader for the frozen persistent V4.12 service bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import pickle
import stat
from types import MappingProxyType
from typing import Any, Mapping

import joblib
import numpy as np
import xgboost as xgb

from . import v411_acceptor as _v411_acceptor
from . import v411_scene as _v411_scene
from . import v49_site_function as _v49_site_function
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
    StrictPartitionError,
    StrictSnapshotLookup,
    StrictTfidfError,
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
    "acceptor_source": (
        Path(
            "/Users/nathanjullia/Documents/Projets/SIRETO/"
            "src/xgb_matcher/v411_acceptor.py"
        ),
        "a85833a78ab98121d3ccf4bc9e93b63cbd33c2e727b980bc456bff64424740b4",
    ),
    "scene_source": (
        Path(
            "/Users/nathanjullia/Documents/Projets/SIRETO/"
            "src/xgb_matcher/v411_scene.py"
        ),
        "ce90f6a9402cd062e5afb1f5248085df551a87f2ac0b40657e7c57b8236086a1",
    ),
    "site_function_source": (
        Path(
            "/Users/nathanjullia/Documents/Projets/SIRETO/"
            "src/xgb_matcher/v49_site_function.py"
        ),
        "8463086d2ce404e5c83140df8ea7351cfb363793edfa7e74db95fe202d9c54e2",
    ),
}
_BUNDLE_ATTESTATION = object()


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


def _path_chain(path: Path) -> tuple[tuple[int, int, int], ...]:
    absolute = Path(path)
    if not absolute.is_absolute():
        _fail(f"frozen path is not absolute: {path}")
    identities: list[tuple[int, int, int]] = []
    current = Path(absolute.anchor)
    root = os.lstat(current)
    if not stat.S_ISDIR(root.st_mode) or stat.S_ISLNK(root.st_mode):
        _fail(f"invalid frozen path root: {path}")
    identities.append((int(root.st_dev), int(root.st_ino), int(root.st_mode)))
    for component in absolute.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            _fail(f"cannot inspect frozen path {current}: errno={exc.errno}")
        if stat.S_ISLNK(metadata.st_mode):
            _fail(f"frozen path contains a symlink: {current}")
        identities.append(
            (
                int(metadata.st_dev),
                int(metadata.st_ino),
                int(metadata.st_mode),
            )
        )
    return tuple(identities)


def _capture_exact(path: Path, expected_sha256: str) -> bytes:
    chain_before = _path_chain(path)
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
        if _path_chain(path) != chain_before:
            _fail(f"frozen path chain changed: {path}")
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
    imported_sources = {
        "acceptor_source": Path(_v411_acceptor.__file__).resolve(),
        "scene_source": Path(_v411_scene.__file__).resolve(),
        "site_function_source": Path(_v49_site_function.__file__).resolve(),
    }
    for role, imported_path in imported_sources.items():
        expected_path = EXPECTED_FILES[role][0].resolve()
        if imported_path != expected_path:
            _fail(
                f"executed source path changed: {role}: {imported_path}"
            )
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
        ("acceptor", "source_sha256"): hashlib.sha256(
            payloads["acceptor_source"]
        ).hexdigest(),
        ("scene", "source_sha256"): hashlib.sha256(
            payloads["scene_source"]
        ).hexdigest(),
        ("scene", "site_function_source_sha256"): hashlib.sha256(
            payloads["site_function_source"]
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


def _validate_runtime(
    certification: Mapping[str, Any],
    ranker_manifest: Mapping[str, Any],
) -> None:
    expected = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "duckdb": importlib.metadata.version("duckdb"),
        "joblib": importlib.metadata.version("joblib"),
        "scipy": importlib.metadata.version("scipy"),
    }
    if certification.get("runtime") != expected:
        _fail("strict-store runtime changed")
    ranker_runtime = ranker_manifest.get("runtime")
    if type(ranker_runtime) is not dict:
        _fail("ranker runtime missing")
    dependencies = ranker_runtime.get("dependencies")
    expected_dependencies = {
        "numpy": expected["numpy"],
        "pandas": expected["pandas"],
        "pyarrow": expected["pyarrow"],
        "scikit-learn": expected["scikit_learn"],
        "xgboost": importlib.metadata.version("xgboost"),
    }
    if (
        ranker_runtime.get("python") != expected["python"]
        or ranker_runtime.get("platform") != expected["platform"]
        or ranker_runtime.get("machine") != expected["machine"]
        or dependencies != expected_dependencies
    ):
        _fail("Ranker C runtime changed")


class ObservedPartitionStore:
    """Expose strict-store misses as measured service telemetry."""

    def __init__(self, inner: StrictPartitionStore) -> None:
        self._inner = inner
        self.sealed_key_miss_count = 0

    @property
    def partition_keys(self):
        return self._inner.partition_keys

    def load(self, partition_key):
        try:
            return self._inner.load(partition_key)
        except StrictPartitionError as exc:
            if "not in the frozen subset" in str(exc):
                self.sealed_key_miss_count += 1
            raise

    def load_with_status(self, partition_key):
        try:
            return self._inner.load_with_status(partition_key)
        except StrictPartitionError as exc:
            if "not in the frozen subset" in str(exc):
                self.sealed_key_miss_count += 1
            raise

    def release(self, partition_key):
        return self._inner.release(partition_key)


class ObservedTfidfCache:
    """Measure sealed misses; rebuild/write are structurally unavailable."""

    rebuild_api_absent = True
    write_api_absent = True

    def __init__(self, inner: StrictVerifiedTfidfCache) -> None:
        self._inner = inner
        self.sealed_key_miss_count = 0
        self.cache_rebuild_count = 0
        self.cache_write_count = 0

    @property
    def partition_keys(self):
        return self._inner.partition_keys

    def get(self, partition_key, aligned_pool):
        try:
            return self._inner.get(partition_key, aligned_pool)
        except StrictTfidfError as exc:
            if "cache miss" in str(exc):
                self.sealed_key_miss_count += 1
            raise

    def release(self, partition_key):
        return self._inner.release(partition_key)


@dataclass(frozen=True)
class FrozenV412ServiceBundle:
    partition_store: ObservedPartitionStore
    tfidf_cache: ObservedTfidfCache
    lookup: StrictSnapshotLookup
    retrieval: V412RetrievalFeatureService
    downstream: V412DownstreamService
    evidence: V412DirectEvidenceService | None
    asset_hashes: Mapping[str, str]
    _attestation: object
    _component_identity: tuple[int, ...]
    _state_identity: tuple[str, ...]
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            self.lookup.close()
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> "FrozenV412ServiceBundle":
        if self._closed:
            _fail("cannot reopen a closed service bundle")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _bundle_state_identity(
    bundle: FrozenV412ServiceBundle,
) -> tuple[str, ...]:
    try:
        ranker_raw = bytes(bundle.downstream.ranker.get_booster().save_raw())
        acceptor_raw = pickle.dumps(
            bundle.downstream.acceptor,
            protocol=5,
        )
        taxonomy_raw = pickle.dumps(
            bundle.downstream.taxonomy,
            protocol=5,
        )
    except Exception as exc:
        _fail(f"cannot fingerprint service model state: {exc}")
    return (
        hashlib.sha256(ranker_raw).hexdigest(),
        hashlib.sha256(acceptor_raw).hexdigest(),
        hashlib.sha256(taxonomy_raw).hexdigest(),
        repr(bundle.downstream.threshold),
        json.dumps(list(bundle.downstream.ranker_feature_order)),
    )


def validate_frozen_v412_service_bundle(
    bundle: FrozenV412ServiceBundle,
) -> None:
    if type(bundle) is not FrozenV412ServiceBundle:
        _fail("unattested or mutated service bundle")
    observed_identity = (
        id(bundle.partition_store),
        id(bundle.partition_store._inner),
        id(bundle.tfidf_cache),
        id(bundle.tfidf_cache._inner),
        id(bundle.lookup),
        id(bundle.retrieval),
        id(bundle.retrieval.retriever),
        id(bundle.retrieval.feature_builder),
        id(bundle.retrieval.idf_builder),
        id(bundle.downstream),
        id(bundle.downstream.ranker),
        id(bundle.downstream.acceptor),
        id(bundle.downstream.taxonomy),
        id(bundle.downstream.scene_builder),
        id(bundle.downstream._trace_origin_token),
        id(bundle.downstream._trace_secret),
        id(bundle.evidence),
        id(bundle.evidence.route) if bundle.evidence is not None else 0,
        (
            id(bundle.evidence.load_partition)
            if bundle.evidence is not None
            else 0
        ),
        id(bundle.evidence.build_index) if bundle.evidence is not None else 0,
        id(bundle.evidence.search) if bundle.evidence is not None else 0,
    )
    expected_hashes = {
        role: digest for role, (_path, digest) in EXPECTED_FILES.items()
    }
    acceptor_classes = np.asarray(
        getattr(bundle.downstream.acceptor, "classes_", []),
    )
    if (
        bundle._attestation is not _BUNDLE_ATTESTATION
        or bundle._closed
        or bundle.retrieval.partition_store is not bundle.partition_store
        or bundle.retrieval.tfidf_cache is not bundle.tfidf_cache
        or bundle.retrieval.lookup is not bundle.lookup
        or bundle._component_identity != observed_identity
        or bundle._state_identity != _bundle_state_identity(bundle)
        or dict(bundle.asset_hashes) != expected_hashes
        or bundle.downstream.ranker_feature_order
        != RANKER_C_FEATURE_ORDER
        or bundle.downstream.threshold != FIXED_THRESHOLD
        or acceptor_classes.shape != (2,)
        or not np.array_equal(acceptor_classes, np.asarray([0, 1]))
        or (
            bundle.evidence is not None
            and bundle.evidence.partition_store is not bundle.partition_store
        )
    ):
        _fail("unattested or mutated service bundle")


def load_frozen_v412_service_bundle(
    *,
    include_evidence: bool,
) -> FrozenV412ServiceBundle:
    """Load the one frozen production stack; no paths or models are injectable."""
    if type(include_evidence) is not bool:
        _fail("include_evidence must be boolean")
    payloads = _capture_bundle_files()
    certification, run_spec, descriptor = _validate_control_files(payloads)
    _validate_model_controls(payloads)
    _validate_runtime(
        certification,
        _json_object(payloads["ranker_manifest"], "ranker manifest"),
    )
    allowed = run_spec.get("allowed_read_files")
    partition_records = run_spec.get("partition_records")
    cache_records = run_spec.get("cache_records")
    if not all(
        isinstance(value, list)
        for value in (allowed, partition_records, cache_records)
    ):
        _fail("strict store records missing")

    partition_store = ObservedPartitionStore(
        StrictPartitionStore(
            partition_records,
            allowed,
            max_cache_entries=5,
        )
    )
    tfidf_cache = ObservedTfidfCache(
        StrictVerifiedTfidfCache(
            cache_records,
            allowed,
            namespace=CACHE_NAMESPACE,
            max_cache_entries=20,
        )
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
        bundle = FrozenV412ServiceBundle(
            partition_store=partition_store,
            tfidf_cache=tfidf_cache,
            lookup=lookup,
            retrieval=retrieval,
            downstream=downstream,
            evidence=evidence,
            asset_hashes=MappingProxyType(
                {
                    role: hashlib.sha256(payload).hexdigest()
                    for role, payload in payloads.items()
                }
            ),
            _attestation=_BUNDLE_ATTESTATION,
            _component_identity=(),
            _state_identity=(),
        )
        object.__setattr__(
            bundle,
            "_component_identity",
            (
                id(bundle.partition_store),
                id(bundle.partition_store._inner),
                id(bundle.tfidf_cache),
                id(bundle.tfidf_cache._inner),
                id(bundle.lookup),
                id(bundle.retrieval),
                id(bundle.retrieval.retriever),
                id(bundle.retrieval.feature_builder),
                id(bundle.retrieval.idf_builder),
                id(bundle.downstream),
                id(bundle.downstream.ranker),
                id(bundle.downstream.acceptor),
                id(bundle.downstream.taxonomy),
                id(bundle.downstream.scene_builder),
                id(bundle.downstream._trace_origin_token),
                id(bundle.downstream._trace_secret),
                id(bundle.evidence),
                id(bundle.evidence.route) if bundle.evidence is not None else 0,
                (
                    id(bundle.evidence.load_partition)
                    if bundle.evidence is not None
                    else 0
                ),
                (
                    id(bundle.evidence.build_index)
                    if bundle.evidence is not None
                    else 0
                ),
                id(bundle.evidence.search) if bundle.evidence is not None else 0,
            ),
        )
        object.__setattr__(
            bundle,
            "_state_identity",
            _bundle_state_identity(bundle),
        )
        validate_frozen_v412_service_bundle(bundle)
        return bundle
    except BaseException:
        lookup.close()
        raise


__all__ = [
    "FrozenV412ServiceBundle",
    "load_frozen_v412_service_bundle",
    "validate_frozen_v412_service_bundle",
]
