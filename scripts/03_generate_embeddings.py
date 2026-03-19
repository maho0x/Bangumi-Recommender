"""
Step 3: Generate LLM embeddings for all 434K subjects.

Uses bge-m3 (BAAI, 1024-dim, supports Chinese/Japanese) to embed
structured text descriptions of each subject.

Outputs:
  - data/embeddings/subject_embeddings.npy  (N × 1024)
  - data/embeddings/subject_ids.json        (index → subject_id mapping)
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EMB_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

TYPE_NAMES = {1: "书籍", 2: "动画", 3: "音乐", 4: "游戏", 6: "三次元"}
BATCH_SIZE = 512  # API batch size - larger = fewer calls
API_BASE = "https://api.studio.nebius.ai/v1/embeddings"
API_KEY = "v1.CmMKHHN0YXRpY2tleS1lMDB5MDBqeHRydjFjMzk1eDgSIXNlcnZpY2VhY2NvdW50LWUwMGg2Z3ZhaGg4OHY4NW5nOTIMCKHi3c0GELDqzJMCOgsIoOX1mAcQwPGjQkACWgNlMDA.AAAAAAAAAAER0vBuzvvbrcPOlKimXwL9PvavUh_v0UvBTTX_69eWClj67A44VH-1YcHbzekrdhgdtuJNT9-mfZeZulZ00U4D"
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


def build_text_description(row):
    """Build a structured text description for embedding."""
    parts = []

    # Type and platform
    type_name = TYPE_NAMES.get(row.get("type"), "未知")
    parts.append(f"[类型: {type_name}]")
    if pd.notna(row.get("platform")) and row["platform"]:
        parts.append(f"[平台: {row['platform']}]")
    if pd.notna(row.get("date")) and row["date"]:
        # Just year-month for brevity
        date_str = str(row["date"])[:7]
        parts.append(f"[日期: {date_str}]")

    # Title
    name_cn = row.get("name_cn", "")
    name = row.get("name", "")
    if name_cn and name and name_cn != name:
        parts.append(f"标题: {name} / {name_cn}")
    elif name_cn:
        parts.append(f"标题: {name_cn}")
    elif name:
        parts.append(f"标题: {name}")

    # Tags
    tags = row.get("tag_list", [])
    if isinstance(tags, str):
        try:
            tags = ast.literal_eval(tags)
        except (ValueError, SyntaxError):
            tags = []
    if not isinstance(tags, list):
        try:
            tags = list(tags)
        except (TypeError, ValueError):
            tags = []
    if len(tags) > 0:
        parts.append(f"标签: {', '.join(str(t) for t in tags[:15])}")

    # Score
    score = row.get("parsed_score") or row.get("score")
    rank = row.get("parsed_rank")
    if score and score > 0:
        score_str = f"评分: {score}"
        if rank and rank > 0:
            score_str += f" (Rank #{rank})"
        parts.append(score_str)

    # Summary from archive if available
    summary = row.get("summary", "")
    if pd.notna(summary) and summary:
        # Truncate long summaries
        if len(summary) > 500:
            summary = summary[:500] + "..."
        parts.append(f"简介: {summary}")

    return "\n".join(parts)


def main():
    import httpx
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Loading subjects metadata ...")
    df = pd.read_parquet(DATA_DIR / "subjects_meta.parquet")
    print(f"  Total subjects: {len(df):,}")

    # Try to merge with archive data for summaries
    archive_path = DATA_DIR / "archive_subjects.parquet"
    if archive_path.exists():
        print("Loading archive data for summaries ...")
        archive = pd.read_parquet(archive_path)
        archive_summary = archive[["id", "summary"]].dropna(subset=["summary"])
        archive_summary = archive_summary[archive_summary["summary"] != ""]
        archive_summary = archive_summary.rename(columns={"summary": "archive_summary"})

        df = df.merge(archive_summary, on="id", how="left")
        df["summary"] = df["archive_summary"].fillna("")
        df = df.drop(columns=["archive_summary"])
        print(f"  Subjects with summaries: {df['summary'].ne('').sum():,}")
    else:
        print("  No archive data found, proceeding without summaries")
        df["summary"] = ""

    # Build text descriptions
    print("Building text descriptions ...")
    texts = []
    subject_ids = []
    for i in tqdm(range(len(df)), desc="Building texts"):
        row = df.iloc[i]
        text = build_text_description(row)
        texts.append(text)
        subject_ids.append(int(row["id"]))

    # Free dataframe memory
    del df
    import gc; gc.collect()

    # Save subject ID mapping
    with open(EMB_DIR / "subject_ids.json", "w") as f:
        json.dump(subject_ids, f)

    print(f"\nUsing API embedding model: {MODEL_NAME}")
    print(f"  API: {API_BASE}")

    # Get embedding dimension with a test call
    test_resp = httpx.post(
        API_BASE,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL_NAME, "input": ["test"]},
        timeout=60,
    )
    test_resp.raise_for_status()
    embed_dim = len(test_resp.json()["data"][0]["embedding"])
    print(f"  Embedding dimension: {embed_dim}")

    n_total = len(texts)
    output_path = EMB_DIR / "subject_embeddings.npy"

    # Create or resume memory-mapped output file
    # np.save header: 128 bytes for .npy format, then data
    # We use a raw binary file and write the .npy at the end
    progress_path = EMB_DIR / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        start_idx = progress["completed_items"]
        print(f"  Resuming from {start_idx:,} items")
    else:
        start_idx = 0

    # Use a flat binary file to append embeddings incrementally
    raw_path = EMB_DIR / "embeddings_raw.bin"
    if start_idx == 0:
        # Fresh start - create new file
        raw_file = open(raw_path, "wb")
    else:
        # Resume - append to existing
        raw_file = open(raw_path, "ab")

    # Build batch ranges from start_idx
    batch_ranges = []
    for i in range(start_idx, n_total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n_total)
        batch_ranges.append((i, end))

    total_batches = len(batch_ranges)
    print(f"Generating embeddings for {n_total - start_idx:,} remaining subjects ({total_batches} batches, {BATCH_SIZE}/batch) ...")

    CONCURRENT = 8

    def embed_batch(batch_texts, batch_idx):
        """Send one batch to the API with retries."""
        for attempt in range(5):
            try:
                resp = httpx.post(
                    API_BASE,
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={"model": MODEL_NAME, "input": batch_texts},
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return batch_idx, np.array([d["embedding"] for d in sorted_data], dtype=np.float32)
            except Exception as e:
                if attempt < 4:
                    _time.sleep(2 ** attempt)
                else:
                    print(f"\n  Batch {batch_idx} failed after 5 attempts: {e}")
                    raise
        return batch_idx, None

    # Process batches and write to disk immediately (in order)
    # We need to buffer out-of-order results but write sequentially
    pending_results = {}
    next_write_batch = 0
    items_written = start_idx
    completed = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT) as executor:
        futures = {}

        # Submit initial work
        for bi in range(min(CONCURRENT, total_batches)):
            i, end = batch_ranges[bi]
            fut = executor.submit(embed_batch, texts[i:end], bi)
            futures[fut] = bi
        next_submit = min(CONCURRENT, total_batches)

        pbar = tqdm(total=total_batches, desc="Embedding")

        while futures:
            for fut in as_completed(futures):
                bi = futures.pop(fut)
                batch_idx, emb = fut.result()
                pending_results[batch_idx] = emb
                completed += 1
                pbar.update(1)

                # Submit next batch
                if next_submit < total_batches:
                    i, end = batch_ranges[next_submit]
                    new_fut = executor.submit(embed_batch, texts[i:end], next_submit)
                    futures[new_fut] = next_submit
                    next_submit += 1

                # Write completed sequential batches to disk
                while next_write_batch in pending_results:
                    batch_emb = pending_results.pop(next_write_batch)
                    batch_emb.tofile(raw_file)
                    items_written += batch_emb.shape[0]
                    next_write_batch += 1

                    # Save progress periodically
                    if next_write_batch % 20 == 0:
                        raw_file.flush()
                        with open(progress_path, "w") as pf:
                            json.dump({"completed_items": items_written}, pf)

                break  # Process one at a time

        pbar.close()

    # Flush remaining
    raw_file.flush()
    raw_file.close()

    print(f"  Total items written: {items_written:,}")

    # Convert raw binary to normalized .npy file using memory mapping
    print("Normalizing embeddings (memory-mapped) ...")
    raw_data = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(items_written, embed_dim))

    # Normalize in chunks to save memory
    CHUNK = 10000
    # Create output .npy with header first
    out_arr = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=np.float32, shape=(items_written, embed_dim)
    )
    for ci in tqdm(range(0, items_written, CHUNK), desc="Normalizing"):
        ce = min(ci + CHUNK, items_written)
        chunk = np.array(raw_data[ci:ce])  # copy to RAM
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1
        out_arr[ci:ce] = chunk / norms
    del out_arr  # flush to disk
    del raw_data

    print(f"  Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Clean up
    raw_path.unlink()
    if progress_path.exists():
        progress_path.unlink()

    # Remove old partial file if exists
    partial_path = EMB_DIR / "subject_embeddings_partial.npy"
    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
