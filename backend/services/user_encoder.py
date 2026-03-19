"""Encode user collections from Bangumi API into formats needed by recommenders."""


def parse_api_collections(raw_collections: list[dict]) -> list[dict]:
    """
    Normalize Bangumi API v0 collection response into a flat format.

    Input format (from API):
    {
        "subject_id": 12345,
        "subject": {"id": 12345, "type": 2, "name": "...", ...},
        "type": 2,  # 1=wish, 2=collect, 3=doing, 4=on_hold, 5=dropped
        "rate": 8,
        "updated_at": "2023-01-01T00:00:00+08:00",
        ...
    }
    """
    parsed = []
    for col in raw_collections:
        subject = col.get("subject", {})
        parsed.append({
            "subject_id": col.get("subject_id", subject.get("id", 0)),
            "subject_type": subject.get("type", 2),
            "type": col.get("type", 2),
            "rate": col.get("rate", 0),
            "updated_at": col.get("updated_at", ""),
            "subject": subject,
        })
    return parsed


def get_collection_stats(collections: list[dict]) -> dict:
    """Compute collection statistics by type."""
    type_counts = {1: 0, 2: 0, 3: 0, 4: 0, 6: 0}
    ratings = []
    tag_counts = {}

    for col in collections:
        st = col.get("subject_type", col.get("subject", {}).get("type", 0))
        if st in type_counts:
            type_counts[st] += 1

        rate = col.get("rate", 0)
        if rate > 0:
            ratings.append(rate)

        tags = col.get("subject", {}).get("tags", [])
        if isinstance(tags, list):
            for tag in tags[:5]:
                name = tag.get("name", "") if isinstance(tag, dict) else str(tag)
                if name:
                    tag_counts[name] = tag_counts.get(name, 0) + 1

    # Top tags
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "anime_count": type_counts.get(2, 0),
        "book_count": type_counts.get(1, 0),
        "music_count": type_counts.get(3, 0),
        "game_count": type_counts.get(4, 0),
        "real_count": type_counts.get(6, 0),
        "avg_rating": sum(ratings) / len(ratings) if ratings else None,
        "top_tags": [{"name": n, "count": c} for n, c in top_tags],
    }
