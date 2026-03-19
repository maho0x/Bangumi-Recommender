"""Collaborative filtering recommender using Multi-VAE."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.config import MODELS_DIR, PROCESSED_DIR


# Inline Multi-VAE model definition (must match training script)
class MultiVAE(torch.nn.Module):
    def __init__(self, n_items, hidden_dim=600, latent_dim=200, dropout=0.5):
        super().__init__()
        self.n_items = n_items
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_items, hidden_dim),
            torch.nn.Tanh(),
        )
        self.mu_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, n_items),
        )
        self.drop = torch.nn.Dropout(dropout)

    def encode(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.drop(x)
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu  # No sampling at inference
        logits = self.decode(z)
        return logits, mu, logvar


# Implicit signal weights
TYPE_WEIGHTS = {
    "wish": 0.6,
    "collect": 1.0,
    "doing": 0.8,
    "on_hold": 0.3,
    "dropped": 0.1,
    1: 0.6,  # API uses integer types
    2: 1.0,
    3: 0.8,
    4: 0.3,
    5: 0.1,
}

# API collection type mapping
COLLECTION_TYPE_MAP = {
    1: "wish",
    2: "collect",
    3: "doing",
    4: "on_hold",
    5: "dropped",
}


class CFRecommender:
    def __init__(self):
        self.model = None
        self.item_id_map = None      # subject_id (str) → contiguous int
        self.item_id_reverse = None  # contiguous int (str) → subject_id
        self.device = torch.device("cpu")
        self.loaded = False

    def load(self):
        """Load trained Multi-VAE model and ID mappings."""
        config_path = MODELS_DIR / "multivae_config.json"
        model_path = MODELS_DIR / "multivae_best.pt"

        if not config_path.exists() or not model_path.exists():
            print("CF model not found, CF recommendations will be unavailable")
            return

        with open(config_path) as f:
            config = json.load(f)

        self.model = MultiVAE(
            n_items=config["n_items"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            dropout=0.0,  # No dropout at inference
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.model.to(self.device)

        with open(PROCESSED_DIR / "item_id_map.json") as f:
            self.item_id_map = json.load(f)
        with open(PROCESSED_DIR / "item_id_reverse.json") as f:
            self.item_id_reverse = json.load(f)

        self.loaded = True
        print(f"CF model loaded: {config['n_items']} items, NDCG={config.get('best_ndcg', 'N/A')}")

    def predict(self, user_collections: list[dict], top_n: int = 100) -> list[tuple[int, float]]:
        """
        Given a user's collections (from Bangumi API), predict scores for all items.

        Args:
            user_collections: List of collection dicts from Bangumi API
            top_n: Number of top predictions to return

        Returns:
            List of (subject_id, score) tuples, sorted by score desc
        """
        if not self.loaded or self.model is None:
            return []

        n_items = self.model.n_items
        interaction_vec = np.zeros(n_items, dtype=np.float32)
        known_items = set()

        for col in user_collections:
            subject_id = str(col.get("subject_id", col.get("subject", {}).get("id", 0)))
            item_idx_str = None

            # Try direct lookup
            if subject_id in self.item_id_map:
                item_idx_str = subject_id
            else:
                continue

            item_idx = self.item_id_map[item_idx_str]
            known_items.add(item_idx)

            # Collection type weight
            col_type = col.get("type", 2)
            type_weight = TYPE_WEIGHTS.get(col_type, 0.5)

            # Rating
            rating = col.get("rate", 0)
            if rating > 0:
                val = type_weight * (rating / 10.0)
            else:
                val = type_weight

            interaction_vec[item_idx] = val

        if not known_items:
            return []

        # Forward pass
        with torch.no_grad():
            x = torch.FloatTensor(interaction_vec).unsqueeze(0).to(self.device)
            logits, _, _ = self.model(x)
            scores = logits.squeeze(0).cpu().numpy()

        # Mask known items
        for idx in known_items:
            scores[idx] = -float("inf")

        # Get top-N
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] == -float("inf"):
                continue
            subject_id = int(self.item_id_reverse.get(str(idx), 0))
            if subject_id > 0:
                # Normalize score to [0, 1] range
                normalized = float(1.0 / (1.0 + np.exp(-scores[idx])))
                results.append((subject_id, normalized))

        return results
