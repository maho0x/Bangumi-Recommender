"""
Step 1b: Download Bangumi Archive from GitHub releases.

The Bangumi Archive provides weekly exports of wiki data including
summaries, infobox, tags, and subject relations for all subject types.

Outputs:
  - data/archive/subject_*.jsonlines (raw archive data)
  - data/processed/archive_subjects.parquet (parsed summaries + relations)
"""

import json
import gzip
import io
from pathlib import Path

import httpx
from tqdm import tqdm

ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "archive"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bangumi Archive GitHub repo
ARCHIVE_REPO = "bangumi/Archive"
ARCHIVE_API = f"https://api.github.com/repos/{ARCHIVE_REPO}/releases/latest"


def get_latest_release_url():
    """Get download URLs from latest Archive release."""
    print("Fetching latest Archive release info ...")
    resp = httpx.get(ARCHIVE_API, timeout=30)
    resp.raise_for_status()
    release = resp.json()
    print(f"  Release: {release['tag_name']} ({release['published_at'][:10]})")

    urls = {}
    for asset in release.get("assets", []):
        name = asset["name"]
        if "subject" in name.lower():
            urls[name] = asset["browser_download_url"]

    return urls


def download_file(url, dest_path):
    """Download a file with progress bar."""
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in resp.iter_bytes(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))


def parse_archive_subjects(archive_path):
    """Parse Archive subject data into a structured format."""
    import pandas as pd

    print(f"Parsing {archive_path.name} ...")
    records = []

    opener = gzip.open if archive_path.suffix == ".gz" else open
    mode = "rt" if archive_path.suffix == ".gz" else "r"

    with opener(archive_path, mode, encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing subjects"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Handle tags - may be list of dicts or list of strings
            raw_tags = obj.get("tags", [])
            if raw_tags and isinstance(raw_tags[0], dict):
                tags_str = json.dumps(raw_tags, ensure_ascii=False)
            else:
                tags_str = json.dumps(raw_tags, ensure_ascii=False)

            record = {
                "id": obj.get("id"),
                "type": obj.get("type"),
                "name": obj.get("name", ""),
                "name_cn": obj.get("name_cn", ""),
                "summary": obj.get("summary", ""),
                "platform": obj.get("platform", ""),
                "date": obj.get("date", ""),
                "nsfw": obj.get("nsfw", False),
                "tags": tags_str,
                "score": obj.get("score", 0),
                "rank": obj.get("rank", 0),
                "total_ratings": obj.get("favorite", {}).get("collect", 0) if isinstance(obj.get("favorite"), dict) else 0,
            }

            # Extract infobox summary if available
            infobox = obj.get("infobox", [])
            if isinstance(infobox, list):
                for item in infobox:
                    if isinstance(item, dict) and item.get("key") in ("简介", "中文名"):
                        if item.get("key") == "中文名" and not record["name_cn"]:
                            val = item.get("value", "")
                            if isinstance(val, str):
                                record["name_cn"] = val

            records.append(record)

    df = pd.DataFrame(records)
    print(f"  Parsed {len(df):,} subjects")

    df.to_parquet(OUT_DIR / "archive_subjects.parquet", index=False)
    print(f"  Saved archive_subjects.parquet")
    return df


def main():
    try:
        urls = get_latest_release_url()
    except Exception as e:
        print(f"Warning: Could not fetch release info: {e}")
        print("Checking for existing archive files ...")
        existing = list(ARCHIVE_DIR.glob("*subject*"))
        if existing:
            print(f"  Found: {[f.name for f in existing]}")
            parse_archive_subjects(existing[0])
            return
        print("  No archive files found. Skipping archive processing.")
        print("  You can manually download from: https://github.com/bangumi/Archive/releases")
        return

    # Download subject data files
    for name, url in urls.items():
        dest = ARCHIVE_DIR / name
        if dest.exists():
            print(f"  {name} already exists, skipping download")
        else:
            print(f"Downloading {name} ...")
            download_file(url, dest)

    # Parse the subject file
    subject_files = list(ARCHIVE_DIR.glob("*subject*"))
    if subject_files:
        parse_archive_subjects(subject_files[0])
    else:
        print("No subject files found after download!")


if __name__ == "__main__":
    main()
