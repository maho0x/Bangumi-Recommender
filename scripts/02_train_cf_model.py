"""
Step 2: Train Multi-VAE collaborative filtering model on bangumi15M data.

Architecture:
  Encoder: n_items → 600 → 200 (μ, logσ²) → z(200-dim)
  Decoder: 200 → 600 → n_items (multinomial softmax)

Outputs:
  - data/models/multivae_best.pt
  - data/models/multivae_config.json
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
HIDDEN_DIM = 600
LATENT_DIM = 200
DROPOUT = 0.5
LR = 1e-3
BATCH_SIZE = 512
EPOCHS = 50
BETA_MAX = 0.2
BETA_ANNEAL_FRAC = 0.8  # Anneal beta over first 80% of epochs
EVAL_EVERY = 5


class SparseDataset(Dataset):
    """Dataset wrapping a sparse matrix (one row per user)."""

    def __init__(self, matrix):
        self.matrix = matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        row = self.matrix[idx].toarray().squeeze().astype(np.float32)
        return torch.from_numpy(row)


class MultiVAE(nn.Module):
    def __init__(self, n_items, hidden_dim=600, latent_dim=200, dropout=0.5):
        super().__init__()
        self.n_items = n_items
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_items, hidden_dim),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_items),
        )

        self.drop = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.drop(x)
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


def loss_function(logits, x, mu, logvar, beta=1.0):
    """Multi-VAE loss: negative multinomial log-likelihood + beta * KL divergence."""
    # Multinomial log-likelihood
    log_softmax = F.log_softmax(logits, dim=1)
    neg_ll = -torch.sum(log_softmax * x, dim=1).mean()

    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return neg_ll + beta * kl, neg_ll, kl


def ndcg_at_k(pred_scores, true_items, k=20):
    """Compute NDCG@K for a batch."""
    batch_size = pred_scores.shape[0]
    _, topk_indices = torch.topk(pred_scores, k, dim=1)

    ndcgs = []
    for i in range(batch_size):
        topk = topk_indices[i].cpu().numpy()
        true = set(true_items[i])
        if not true:
            continue

        dcg = 0.0
        for rank, item in enumerate(topk):
            if item in true:
                dcg += 1.0 / np.log2(rank + 2)

        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(true), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(ndcgs) if ndcgs else 0.0


def recall_at_k(pred_scores, true_items, k=20):
    """Compute Recall@K for a batch."""
    batch_size = pred_scores.shape[0]
    _, topk_indices = torch.topk(pred_scores, k, dim=1)

    recalls = []
    for i in range(batch_size):
        topk = set(topk_indices[i].cpu().numpy())
        true = set(true_items[i])
        if not true:
            continue
        recalls.append(len(topk & true) / len(true))

    return np.mean(recalls) if recalls else 0.0


def evaluate(model, train_matrix, test_matrix, device, k=20):
    """Evaluate on held-out test set."""
    model.eval()
    ndcgs, recalls = [], []

    n_users = train_matrix.shape[0]
    batch_size = 512

    with torch.no_grad():
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            train_batch = torch.FloatTensor(
                train_matrix[start:end].toarray()
            ).to(device)
            test_batch = test_matrix[start:end]

            # Get predictions
            logits, _, _ = model(train_batch)

            # Mask out training items
            logits[train_batch > 0] = -float("inf")

            # Get true test items for each user
            true_items = []
            for i in range(end - start):
                row = test_batch[i]
                items = row.nonzero()[1].tolist() if sparse.issparse(row) else []
                true_items.append(items)

            # Skip users with no test items
            valid_mask = [len(t) > 0 for t in true_items]
            if not any(valid_mask):
                continue

            ndcgs.append(ndcg_at_k(logits, true_items, k))
            recalls.append(recall_at_k(logits, true_items, k))

    return np.mean(ndcgs), np.mean(recalls)


def create_train_test_split(matrix, test_frac=0.1, min_interactions=5):
    """Hold out test_frac of interactions per user for evaluation (vectorized)."""
    coo = matrix.tocoo()
    rng = np.random.RandomState(42)

    # Group by user
    train_mask = np.ones(coo.nnz, dtype=bool)

    # Sort by user for efficient grouping
    order = np.argsort(coo.row)
    rows = coo.row[order]
    cols = coo.col[order]
    data = coo.data[order]

    # Find user boundaries
    user_starts = np.searchsorted(rows, np.arange(matrix.shape[0]))
    user_ends = np.searchsorted(rows, np.arange(matrix.shape[0]), side='right')

    for u in range(matrix.shape[0]):
        start, end = user_starts[u], user_ends[u]
        n_items = end - start
        if n_items < min_interactions:
            continue
        n_test = max(1, int(n_items * test_frac))
        test_indices = rng.choice(n_items, size=n_test, replace=False)
        train_mask[start + test_indices] = False

    test_mask = ~train_mask

    train = sparse.csr_matrix(
        (data[train_mask], (rows[train_mask], cols[train_mask])),
        shape=matrix.shape,
    )
    test = sparse.csr_matrix(
        (data[test_mask], (rows[test_mask], cols[test_mask])),
        shape=matrix.shape,
    )

    return train, test


def main():
    print("Loading interaction matrix ...")
    matrix = sparse.load_npz(DATA_DIR / "interaction_matrix.npz")
    n_users, n_items = matrix.shape
    print(f"  Shape: {n_users:,} users × {n_items:,} items, {matrix.nnz:,} interactions")

    # Train/test split
    print("Creating train/test split ...")
    train_matrix, test_matrix = create_train_test_split(matrix)
    print(f"  Train: {train_matrix.nnz:,}, Test: {test_matrix.nnz:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Build model
    model = MultiVAE(n_items, HIDDEN_DIM, LATENT_DIM, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Training
    dataset = SparseDataset(train_matrix)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best_ndcg = 0.0
    total_steps = EPOCHS * len(loader)
    anneal_steps = int(total_steps * BETA_ANNEAL_FRAC)

    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_kl = 0.0
        n_batches = 0

        t0 = time.time()
        for batch in loader:
            batch = batch.to(device)

            # Beta annealing
            beta = min(BETA_MAX, BETA_MAX * step / anneal_steps) if anneal_steps > 0 else BETA_MAX

            logits, mu, logvar = model(batch)
            loss, nll, kl = loss_function(logits, batch, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_kl += kl.item()
            n_batches += 1
            step += 1

        avg_loss = epoch_loss / n_batches
        avg_nll = epoch_nll / n_batches
        avg_kl = epoch_kl / n_batches
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} (NLL: {avg_nll:.4f}, KL: {avg_kl:.4f}) | "
            f"Beta: {beta:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Evaluate periodically
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            ndcg, recall = evaluate(model, train_matrix, test_matrix, device)
            print(f"  → NDCG@20: {ndcg:.4f}, Recall@20: {recall:.4f}")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                torch.save(model.state_dict(), MODEL_DIR / "multivae_best.pt")
                print(f"  → New best! Saved model.")

    # Save config
    config = {
        "n_items": n_items,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "dropout": DROPOUT,
        "best_ndcg": float(best_ndcg),
    }
    with open(MODEL_DIR / "multivae_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete! Best NDCG@20: {best_ndcg:.4f}")
    print(f"Model saved to {MODEL_DIR / 'multivae_best.pt'}")


if __name__ == "__main__":
    main()
