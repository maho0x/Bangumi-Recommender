"""
Step 1: Clean bangumi15M data, build interaction matrix and ID mappings.

Outputs:
  - data/processed/user_id_map.json      (anonymous_id -> contiguous int)
  - data/processed/item_id_map.json      (subject_id -> contiguous int)
  - data/processed/item_id_reverse.json  (contiguous int -> subject_id)
  - data/processed/interaction_matrix.npz (sparse CSR, users × items)
  - data/processed/subjects_meta.parquet (all 434K subjects metadata)
"""

import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "bangumi15M_data" / "raw_data"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Implicit signal weights for collection types
TYPE_WEIGHTS = {
    "collect": 1.0,
    "doing": 0.8,
    "wish": 0.6,
    "on_hold": 0.3,
    "dropped": 0.1,
}


def load_and_clean_interactions():
    print("Loading AnonymousUserCollection.csv ...")
    df = pd.read_csv(RAW_DIR / "AnonymousUserCollection.csv")
    print(f"  Raw rows: {len(df):,}")

    # Drop exact duplicates
    df = df.drop_duplicates()
    print(f"  After dedup: {len(df):,}")

    # Keep only anime (subject_type=2) — all rows should be type 2 but be safe
    df = df[df["subject_type"] == 2].copy()
    print(f"  Anime rows: {len(df):,}")

    # If a user has multiple entries for the same subject, keep the latest
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df = df.sort_values("updated_at").drop_duplicates(
        subset=["user_id", "subject_id"], keep="last"
    )
    print(f"  After user-item dedup: {len(df):,}")

    return df


def build_id_mappings(df):
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["subject_id"].unique())

    user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_map = {int(sid): idx for idx, sid in enumerate(unique_items)}
    item_reverse = {idx: int(sid) for sid, idx in item_map.items()}

    print(f"  Users: {len(user_map):,}, Items: {len(item_map):,}")

    with open(OUT_DIR / "user_id_map.json", "w") as f:
        json.dump(user_map, f)
    with open(OUT_DIR / "item_id_map.json", "w") as f:
        json.dump(item_map, f)
    with open(OUT_DIR / "item_id_reverse.json", "w") as f:
        json.dump(item_reverse, f)

    return user_map, item_map


def build_interaction_matrix(df, user_map, item_map):
    n_users = len(user_map)
    n_items = len(item_map)
    print(f"Building interaction matrix ({n_users} x {n_items}) ...")

    # Vectorized: map user_id and subject_id to contiguous indices
    print("  Mapping IDs ...")
    rows = df["user_id"].map(user_map).values
    cols = df["subject_id"].map(item_map).values

    # Vectorized: compute interaction values
    print("  Computing interaction values ...")
    type_weights = df["type"].map(TYPE_WEIGHTS).fillna(0.5).values
    ratings = df["rating"].values

    # Where rating > 0: type_weight * (rating / 10), else: type_weight
    vals = np.where(ratings > 0, type_weights * (ratings / 10.0), type_weights).astype(np.float32)

    mat = sparse.csr_matrix(
        (vals, (rows.astype(np.int32), cols.astype(np.int32))),
        shape=(n_users, n_items),
    )

    sparse.save_npz(OUT_DIR / "interaction_matrix.npz", mat)
    print(f"  Matrix saved: {mat.nnz:,} non-zeros")
    return mat


def process_subjects():
    print("Loading Subjects.csv ...")
    df = pd.read_csv(RAW_DIR / "Subjects.csv", index_col=0)
    print(f"  Total subjects: {len(df):,}")

    # Parse tags from string representation to list of tag names
    def extract_tag_names(tags_str):
        if pd.isna(tags_str):
            return []
        try:
            tags = ast.literal_eval(tags_str)
            return [t["name"] for t in tags[:20]]  # Keep top 20 tags
        except (ValueError, SyntaxError):
            return []

    # Parse rating dict to extract score and rank
    def extract_rating_info(rating_str):
        if pd.isna(rating_str):
            return None, None
        try:
            r = ast.literal_eval(rating_str)
            return r.get("score"), r.get("rank")
        except (ValueError, SyntaxError):
            return None, None

    print("Parsing tags ...")
    df["tag_list"] = df["tags"].apply(extract_tag_names)

    print("Parsing rating info ...")
    rating_info = df["rating"].apply(extract_rating_info)
    df["parsed_score"] = rating_info.apply(lambda x: x[0])
    df["parsed_rank"] = rating_info.apply(lambda x: x[1])

    # Save as parquet
    # Select and rename columns for clarity
    out_cols = [
        "id", "name_cn", "name", "date", "type", "score",
        "on_hold", "dropped", "wish", "collect", "doing",
        "platform", "tag_list", "total_episodes", "eps", "volumes",
        "locked", "nsfw", "parsed_score", "parsed_rank",
    ]
    df_out = df[out_cols].copy()
    df_out.to_parquet(OUT_DIR / "subjects_meta.parquet")
    print(f"  Saved subjects_meta.parquet ({len(df_out):,} rows)")

    # Print type distribution
    type_names = {1: "书籍", 2: "动画", 3: "音乐", 4: "游戏", 6: "三次元"}
    print("\n  Type distribution:")
    for t, name in type_names.items():
        count = (df["type"] == t).sum()
        print(f"    {name} (type={t}): {count:,}")


def main():
    df = load_and_clean_interactions()
    user_map, item_map = build_id_mappings(df)
    build_interaction_matrix(df, user_map, item_map)
    process_subjects()
    print("\nDone! All outputs saved to", OUT_DIR)


if __name__ == "__main__":
    main()
