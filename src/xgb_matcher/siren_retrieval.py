"""
SIREN-level retrieval for Route B architecture.

Provides fast lookup of:
1. Top-K SIRENs by name similarity (via global TF-IDF index)
2. Geographic locations for each SIREN (for SIRET retrieval)
"""

from pathlib import Path
from typing import Dict, List, Tuple
import logging

import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class SirenGlobalIndex:
    """Query interface for global SIREN TF-IDF index (7M SIRENs).

    Uses two-channel retrieval pattern:
    - Primary: word n-gram TF-IDF (high precision)
    - Fallback: char n-gram TF-IDF (handles acronyms, typos)

    Both matrices are L2-normalized → dot product = cosine similarity.
    """

    def __init__(self, index_dir: Path):
        """Load pre-built SIREN index artifacts.

        Args:
            index_dir: Directory containing vectorizers and matrices from build_siren_global_index.py
        """
        index_dir = Path(index_dir)

        logger.info(f"Loading SIREN global index from {index_dir}...")

        self.word_vec = joblib.load(index_dir / "word_vectorizer.pkl")
        self.char_vec = joblib.load(index_dir / "char_vectorizer.pkl")

        # Load matrices (L2-normalized CSR)
        self.word_mat = scipy.sparse.load_npz(index_dir / "word_matrix.npz").tocsr()
        self.char_mat = scipy.sparse.load_npz(index_dir / "char_matrix.npz").tocsr()

        # Load SIREN IDs (7M array)
        self.siren_ids = np.load(index_dir / "siren_ids.npy", allow_pickle=True)

        logger.info(
            f"Loaded: {len(self.siren_ids)} SIRENs, "
            f"word matrix {self.word_mat.shape}, char matrix {self.char_mat.shape}"
        )

    def query(
        self,
        crm_name_bag: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Query top-K SIRENs by name similarity.

        Args:
            crm_name_bag: Normalized CRM name string (space-separated tokens)
            top_k: Number of top SIRENs to return (default 50)

        Returns:
            List of (siren_str, cosine_similarity_score) tuples, sorted descending by score.
        """
        if not crm_name_bag or not crm_name_bag.strip():
            return []

        # === Word TF-IDF channel (primary) ===
        q_word = normalize(
            self.word_vec.transform([crm_name_bag]),
            norm='l2'
        )
        row_w = (q_word @ self.word_mat.T).getrow(0)  # sparse (1 × 7M)

        # Extract (index, score) from sparse row (avoid .toarray())
        word_scores: Dict[int, float] = dict(
            zip(row_w.indices.tolist(), row_w.data.tolist())
        )

        # === Char TF-IDF channel (fallback for acronyms/typos) ===
        q_char = normalize(
            self.char_vec.transform([crm_name_bag]),
            norm='l2'
        )
        row_c = (q_char @ self.char_mat.T).getrow(0)  # sparse

        char_scores: Dict[int, float] = dict(
            zip(row_c.indices.tolist(), row_c.data.tolist())
        )

        # === Union with word priority ===
        # Word scores as primary; char scores (0.8 weight) fill gaps
        all_scores = {
            **{i: s * 0.8 for i, s in char_scores.items()},  # char fallback
            **word_scores,  # word overrides
        }

        if not all_scores:
            return []

        # === Top-K selection via argpartition (O(n), not O(n log n)) ===
        idxs = np.array(list(all_scores.keys()), dtype=np.int32)
        vals = np.array(list(all_scores.values()), dtype=np.float32)

        if len(idxs) > top_k:
            # argpartition: O(n) selection of top-k indices
            sel = np.argpartition(vals, -top_k)[-top_k:]
            idxs, vals = idxs[sel], vals[sel]

        # Sort descending by score
        order = np.argsort(vals)[::-1]
        return [
            (self.siren_ids[idxs[i]], float(vals[i]))
            for i in order
        ]


class SirenToGeoIndex:
    """Mapping from SIREN → geographic locations (INSEE, postcode).

    Used to prioritize SIRET retrieval for a given SIREN:
    - First try locations matching CRM INSEE
    - Then locations matching CRM postcode
    - Fall back to all known locations
    """

    def __init__(self, path: Path):
        """Load siren_to_geo index from parquet.

        Args:
            path: Path to siren_to_geo.parquet (columns: siren, insee, postcode, siret_count)
        """
        path = Path(path)
        logger.info(f"Loading siren_to_geo index from {path}...")

        df = pd.read_parquet(path)  # siren, insee, postcode, siret_count

        # Build dict: siren → sorted list of (insee, postcode, siret_count)
        # Sorted by siret_count descending (large geo areas first)
        self._index: Dict[str, List[Tuple[str, str, int]]] = (
            df.sort_values("siret_count", ascending=False)
            .groupby("siren")
            .apply(
                lambda g: list(zip(g["insee"].values, g["postcode"].values, g["siret_count"].values))
            )
            .to_dict()
        )

        logger.info(f"Loaded {len(self._index)} unique SIRENs with geo info")

    def get_locations(self, siren: str) -> List[Tuple[str, str]]:
        """Get all geographic locations (insee, postcode) for a SIREN.

        Returns locations sorted by siret_count descending (largest areas first).

        Args:
            siren: SIREN string

        Returns:
            List of (insee, postcode) tuples, or [] if SIREN not found.
        """
        locations = self._index.get(siren, [])
        return [(r[0], r[1]) for r in locations]

    def get_location_counts(self, siren: str) -> List[Tuple[str, str, int]]:
        """Get locations with SIRET counts (for debugging).

        Args:
            siren: SIREN string

        Returns:
            List of (insee, postcode, siret_count) tuples.
        """
        return self._index.get(siren, [])
