"""Content-based recommender using FAISS nearest-neighbor search on LLM embeddings."""

import json
import math
from pathlib import Path

import faiss
import numpy as np

from backend.config import EMBEDDINGS_DIR, SUBJECT_TYPES

# Collection type weights (same as CF)
TYPE_WEIGHTS = {1: 0.6, 2: 1.0, 3: 0.8, 4: 0.3, 5: 0.1}

# Time decay half-life in days
TIME_DECAY_HALF_LIFE = 365 * 2  # 2 years


class ContentRecommender:
    def __init__(self):
        self.indices: dict[int, faiss.Index] = {}
        self.id_maps: dict[int, list[int]] = {}
        self.embeddings: np.ndarray | None = None
        self.pca_matrix: np.ndarray | None = None   # (PCA_DIM, 4096)
        self.pca_mean: np.ndarray | None = None     # (4096,)
        self.subject_id_to_emb_idx: dict[int, int] = {}
        self.loaded = False

    def load(self):
        """Load FAISS indices and embedding data."""
        emb_path = EMBEDDINGS_DIR / "subject_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / "subject_ids.json"
        pca_matrix_path = EMBEDDINGS_DIR / "pca_matrix.npy"
        pca_mean_path = EMBEDDINGS_DIR / "pca_mean.npy"

        if not emb_path.exists() or not ids_path.exists():
            print("Embeddings not found, content recommendations will be unavailable")
            return

        self.embeddings = np.load(emb_path, mmap_mode="r")
        with open(ids_path) as f:
            subject_ids = json.load(f)

        self.subject_id_to_emb_idx = {sid: idx for idx, sid in enumerate(subject_ids)}

        if pca_matrix_path.exists() and pca_mean_path.exists():
            self.pca_matrix = np.load(pca_matrix_path)  # (PCA_DIM, 4096)
            self.pca_mean = np.load(pca_mean_path)       # (4096,)
            print(f"Loaded {len(subject_ids):,} subject embeddings ({self.embeddings.shape[1]}-dim → {self.pca_matrix.shape[0]}-dim PCA)")
        else:
            print(f"Loaded {len(subject_ids):,} subject embeddings ({self.embeddings.shape[1]}-dim, no PCA)")

        # Load per-type FAISS indices
        for type_id in SUBJECT_TYPES:
            index_path = EMBEDDINGS_DIR / f"faiss_index_type{type_id}.bin"
            id_map_path = EMBEDDINGS_DIR / f"faiss_id_map_type{type_id}.json"

            if index_path.exists() and id_map_path.exists():
                self.indices[type_id] = faiss.read_index(str(index_path))
                with open(id_map_path) as f:
                    self.id_maps[type_id] = json.load(f)
                print(f"  FAISS index type={type_id}: {len(self.id_maps[type_id]):,} items")

        self.loaded = True

    def _build_user_profile(self, user_collections: list[dict]) -> np.ndarray | None:
        """Build a user preference vector by weighted average of collected item embeddings."""
        if self.embeddings is None:
            return None

        weights = []
        emb_indices = []

        for col in user_collections:
            subject_id = col.get("subject_id", col.get("subject", {}).get("id", 0))
            emb_idx = self.subject_id_to_emb_idx.get(subject_id)
            if emb_idx is None:
                continue

            # Type weight
            col_type = col.get("type", 2)
            w = TYPE_WEIGHTS.get(col_type, 0.5)

            # Rating boost
            rating = col.get("rate", 0)
            if rating > 0:
                w *= rating / 10.0

            # Time decay (optional, based on updated_at)
            updated_at = col.get("updated_at", "")
            if updated_at:
                try:
                    from datetime import datetime, timezone
                    if isinstance(updated_at, str):
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    else:
                        dt = updated_at
                    days_ago = (datetime.now(timezone.utc) - dt).days
                    decay = math.exp(-0.693 * days_ago / TIME_DECAY_HALF_LIFE)
                    w *= decay
                except (ValueError, TypeError):
                    pass

            weights.append(w)
            emb_indices.append(emb_idx)

        if not weights:
            return None

        # Weighted average
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()

        profile = np.zeros(self.embeddings.shape[1], dtype=np.float32)
        for w, idx in zip(weights, emb_indices):
            profile += w * np.array(self.embeddings[idx], dtype=np.float32)

        # Project through PCA if available (must match index space)
        if self.pca_matrix is not None and self.pca_mean is not None:
            profile = (profile - self.pca_mean) @ self.pca_matrix.T

        # Normalize
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm

        return profile

    def recommend(
        self,
        user_collections: list[dict],
        subject_type: int = 2,
        top_n: int = 100,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Generate content-based recommendations.

        Returns:
            List of (subject_id, similarity_score) tuples
        """
        if not self.loaded or subject_type not in self.indices:
            return []

        profile = self._build_user_profile(user_collections)
        if profile is None:
            return []

        index = self.indices[subject_type]
        id_map = self.id_maps[subject_type]

        # Search more than needed to account for filtering
        k = min(top_n * 3, len(id_map))
        query = profile.reshape(1, -1)
        distances, indices = index.search(query, k)

        results = []
        exclude = exclude_ids or set()

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(id_map):
                continue
            subject_id = id_map[idx]
            if subject_id in exclude:
                continue

            # Inner product similarity (embeddings are normalized, so in [0,1])
            score = float(max(0, min(1, (dist + 1) / 2)))  # Map [-1,1] to [0,1]
            results.append((subject_id, score))

            if len(results) >= top_n:
                break

        return results

    def get_embedding(self, subject_id: int) -> np.ndarray | None:
        """Get the PCA-projected embedding vector for a subject."""
        idx = self.subject_id_to_emb_idx.get(subject_id)
        if idx is None or self.embeddings is None:
            return None
        vec = np.array(self.embeddings[idx], dtype=np.float32)
        if self.pca_matrix is not None and self.pca_mean is not None:
            vec = (vec - self.pca_mean) @ self.pca_matrix.T
        return vec
