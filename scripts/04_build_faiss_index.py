"""
Step 4: Build FAISS indices per subject type for fast nearest-neighbor search.

PCA reduces 4096-dim embeddings to PCA_DIM (512) for memory efficiency.
The PCA matrix is saved so the backend can project user profiles consistently.

Indices:
  - anime (type=2, ~23K): IndexFlatIP (brute force, <1ms)
  - books (type=1, ~267K): IndexIVFFlat (256 clusters)
  - music (type=3), games (type=4), real (type=6): IndexFlatIP

Outputs:
  - data/embeddings/pca_matrix.npy          (4096 → PCA_DIM projection)
  - data/embeddings/pca_mean.npy            (mean vector for centering)
  - data/embeddings/faiss_index_type{N}.bin
  - data/embeddings/faiss_id_map_type{N}.json  (faiss_idx → subject_id)
"""

import gc
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

EMB_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TYPE_NAMES = {1: "书籍", 2: "动画", 3: "音乐", 4: "游戏", 6: "三次元"}
IVF_THRESHOLD = 50000  # Use IVF for types with more than this many items
IVF_NLIST = 256
PCA_DIM = 512          # Reduce from 4096 to this (16× compression)
PCA_SAMPLE = 100000    # Samples for PCA fitting


def fit_pca(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA using FAISS PCAMatrix on a random sample."""
    n = embeddings.shape[0]
    orig_dim = embeddings.shape[1]
    sample_idx = np.random.choice(n, min(PCA_SAMPLE, n), replace=False)

    print(f"Fitting PCA {orig_dim} → {PCA_DIM} on {len(sample_idx):,} samples ...")
    sample = np.array(embeddings[sample_idx], dtype=np.float32)

    mean = sample.mean(axis=0)
    sample -= mean

    pca = faiss.PCAMatrix(orig_dim, PCA_DIM)
    pca.train(sample)
    A = faiss.vector_float_to_array(pca.A).reshape(PCA_DIM, orig_dim)
    return A, mean


def project_chunked(embeddings: np.ndarray, A: np.ndarray, mean: np.ndarray,
                    indices: list[int], chunk_size: int = 5000) -> np.ndarray:
    """Project selected rows of mmap'd embeddings through PCA in chunks."""
    n = len(indices)
    out = np.empty((n, PCA_DIM), dtype=np.float32)
    for start in tqdm(range(0, n, chunk_size), desc="  Projecting", leave=False):
        end = min(start + chunk_size, n)
        batch_idx = indices[start:end]
        batch = np.array(embeddings[batch_idx], dtype=np.float32) - mean
        out[start:end] = batch @ A.T
    # L2-normalize after PCA so inner product = cosine similarity
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    out /= norms
    return out


def main():
    print("Loading embeddings and metadata ...")
    embeddings = np.load(EMB_DIR / "subject_embeddings.npy", mmap_mode="r")
    with open(EMB_DIR / "subject_ids.json") as f:
        subject_ids = json.load(f)

    df = pd.read_parquet(DATA_DIR / "subjects_meta.parquet")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Subject IDs: {len(subject_ids)}")

    sid_to_emb_idx = {sid: idx for idx, sid in enumerate(subject_ids)}

    # Fit PCA on full corpus
    A, mean = fit_pca(embeddings)
    np.save(EMB_DIR / "pca_matrix.npy", A)
    np.save(EMB_DIR / "pca_mean.npy", mean)
    print(f"  PCA saved: {PCA_DIM}-dim")

    # Build per-type indices
    for type_id, type_name in TYPE_NAMES.items():
        type_sids = df[df["type"] == type_id]["id"].tolist()
        valid_indices = []
        valid_sids = []
        for sid in type_sids:
            emb_idx = sid_to_emb_idx.get(sid)
            if emb_idx is not None:
                valid_indices.append(emb_idx)
                valid_sids.append(int(sid))

        if not valid_indices:
            print(f"\n  {type_name} (type={type_id}): No items, skipping")
            continue

        n_items = len(valid_sids)
        print(f"\n  {type_name} (type={type_id}): {n_items:,} items")

        # Project to PCA space in chunks
        type_embeddings = project_chunked(embeddings, A, mean, valid_indices)
        dim = type_embeddings.shape[1]  # == PCA_DIM

        # Choose index type
        if n_items > IVF_THRESHOLD:
            nlist = min(IVF_NLIST, n_items // 40)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(type_embeddings)
            index.add(type_embeddings)
            index.nprobe = 16
            print(f"    IndexIVFFlat, nlist={nlist}, nprobe=16")
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(type_embeddings)
            print(f"    IndexFlatIP (brute force)")

        index_path = EMB_DIR / f"faiss_index_type{type_id}.bin"
        faiss.write_index(index, str(index_path))
        print(f"    Saved: {index_path.name} ({index_path.stat().st_size / 1024 / 1024:.1f} MB)")

        id_map_path = EMB_DIR / f"faiss_id_map_type{type_id}.json"
        with open(id_map_path, "w") as f:
            json.dump(valid_sids, f)
        print(f"    Saved: {id_map_path.name}")

        del type_embeddings, index
        gc.collect()


if __name__ == "__main__":
    main()
