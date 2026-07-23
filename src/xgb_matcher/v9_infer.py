"""Single-ranker + query-acceptor V9 inference engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .candidates import set_global_name_idf_map
from .dense_retrieval import GlobalDenseSirenIndex, PartitionEmbeddingStore
from .features import (
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
)
from .partitioned_store import PartitionedCandidateStore
from .retrieval import build_candidate_pool
from .retrieval_config import RetrievalConfigV1
from .siren_retrieval import SirenToGeoIndex
from .semantic import set_semantic_client
from .semantic_process import SemanticProcessClient
from .v9_acceptor import V9AcceptorBundle
from .v9_dataset import V9DatasetManifest
from .v9_features import inject_retrieval_siren_features
from .v9_scene import build_inference_scene


class V9InferenceEngine:
    def __init__(
        self,
        *,
        store: PartitionedCandidateStore,
        retrieval_config: RetrievalConfigV1,
        ranker: xgb.XGBRanker,
        ranker_feature_order: list[str],
        acceptor: V9AcceptorBundle,
        dense_store: PartitionEmbeddingStore | None = None,
        dense_siren_index: GlobalDenseSirenIndex | None = None,
        siren_to_geo: SirenToGeoIndex | None = None,
        semantic_client: SemanticProcessClient | None = None,
    ) -> None:
        self.store = store
        self.retrieval_config = retrieval_config
        self.ranker = ranker
        self.ranker_feature_order = ranker_feature_order
        self.acceptor = acceptor
        self.dense_store = dense_store
        self.dense_siren_index = dense_siren_index
        self.siren_to_geo = siren_to_geo
        self.semantic_client = semantic_client
        self._tfidf_cache: dict = {}

    @classmethod
    def from_bundle(
        cls,
        *,
        dataset_dir: Path,
        ranker_dir: Path,
        acceptor_dir: Path,
        partitions_dir: Path,
        retrieval_config: RetrievalConfigV1,
        dense_store_dir: Path | None = None,
        semantic_model_path: Path | None = Path(
            "models/semantic/siret-bert-deploy"
        ),
    ) -> "V9InferenceEngine":
        manifest = V9DatasetManifest.load(dataset_dir / "manifest.json")
        manifest.validate(
            retrieval_config=retrieval_config,
            feature_order=manifest.feature_order,
        )
        ranker_metadata = json.loads(
            (ranker_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if ranker_metadata["dataset_manifest_id"] != manifest.build_id:
            raise ValueError("Ranker/dataset manifest mismatch")
        acceptor = V9AcceptorBundle.load(acceptor_dir)
        if acceptor.dataset_manifest_id != manifest.build_id:
            raise ValueError("Acceptor/dataset manifest mismatch")

        ranker = xgb.XGBRanker()
        ranker.load_model(ranker_dir / "ranker.json")
        dense_store = (
            PartitionEmbeddingStore(dense_store_dir) if dense_store_dir else None
        )
        dense_siren_index = None
        siren_to_geo = None
        semantic_client = None
        if semantic_model_path is not None:
            semantic_client = SemanticProcessClient(
                semantic_model_path,
                device="cpu",
            )
            set_semantic_client(semantic_client)
        if retrieval_config.global_dense_siren_enabled:
            if (
                not retrieval_config.global_dense_siren_index_path
                or not retrieval_config.siren_geo_index_path
            ):
                raise ValueError("Global dense SIREN retrieval requires both index paths")
            dense_siren_index = GlobalDenseSirenIndex(
                Path(retrieval_config.global_dense_siren_index_path)
            )
            index_tokenizer = dense_siren_index.manifest.get("tokenizer_fingerprint")
            if (
                manifest.tokenizer_fingerprint
                and index_tokenizer
                and index_tokenizer != manifest.tokenizer_fingerprint
            ):
                raise ValueError("Global dense SIREN tokenizer fingerprint mismatch")
            siren_to_geo = SirenToGeoIndex(
                Path(retrieval_config.siren_geo_index_path)
            )
        return cls(
            store=PartitionedCandidateStore(partitions_dir),
            retrieval_config=retrieval_config,
            ranker=ranker,
            ranker_feature_order=list(ranker_metadata["feature_order"]),
            acceptor=acceptor,
            dense_store=dense_store,
            dense_siren_index=dense_siren_index,
            siren_to_geo=siren_to_geo,
            semantic_client=semantic_client,
        )

    def close(self) -> None:
        if self.dense_store is not None:
            self.dense_store.close()
        if self.dense_siren_index is not None:
            self.dense_siren_index.close()
        if self.semantic_client is not None:
            self.semantic_client.close()
            self.semantic_client = None
            set_semantic_client(None)

    def infer(self, crm_row: dict[str, Any]):
        query_id = str(crm_row.get("query_id") or crm_row.get("crm_id") or "")
        crm_pre = preprocess_crm_row(crm_row)
        pool = build_candidate_pool(
            self.store,
            crm_row,
            crm_pre,
            self.retrieval_config,
            self._tfidf_cache,
            dense_store=self.dense_store,
            dense_siren_index=self.dense_siren_index,
            siren_to_geo=self.siren_to_geo,
        )
        candidates = pool.candidates
        if not candidates:
            return self.acceptor.decide(
                build_inference_scene(query_id, pd.DataFrame())
            )

        set_global_name_idf_map(pool.idf_map, pool.default_idf)
        feature_rows = make_feature_rows_from_preprocessed(
            crm_pre,
            candidates,
            include_semantic=True,
        )
        inject_retrieval_siren_features(feature_rows, candidates)
        matrix = pd.DataFrame(feature_rows)[self.ranker_feature_order].astype(float)
        scores = self.ranker.predict(matrix.to_numpy())
        predictions = pd.DataFrame(
            {
                "query_id": query_id,
                "candidate_siret": [
                    str(candidate.get("siret") or "") for candidate in candidates
                ],
                "score": np.asarray(scores, dtype=float),
                "sparse_rank": [candidate.get("sparse_rank") for candidate in candidates],
                "dense_rank": [candidate.get("dense_rank") for candidate in candidates],
                "rrf_score": [candidate.get("rrf_score") for candidate in candidates],
                "retrieval_channel_count": [
                    candidate.get("retrieval_channel_count") for candidate in candidates
                ],
                "retrieval_agreement": [
                    candidate.get("retrieval_agreement") for candidate in candidates
                ],
            }
        )
        predictions["rank"] = (
            predictions["score"].rank(method="first", ascending=False).astype(int)
        )
        scene = build_inference_scene(query_id, predictions)
        return self.acceptor.decide(scene)


__all__ = ["V9InferenceEngine"]
