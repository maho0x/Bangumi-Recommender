"""Recommendation API endpoints."""

import json
import time

from fastapi import APIRouter, HTTPException, Query

from backend import deps
from backend.config import (
    CACHE_DB_PATH,
    CACHE_TTL_RECOMMEND_RESULT,
    DEFAULT_LIMIT,
    SUBJECT_TYPES,
)
from backend.models.schemas import (
    HealthResponse,
    RecommendationItem,
    RecommendResponse,
    UserProfile,
)
from backend.services.user_encoder import get_collection_stats, parse_api_collections

router = APIRouter(prefix="/api", tags=["recommend"])


def _cache_get_recommend(key: str):
    """Check recommendation cache."""
    import sqlite3
    try:
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        row = conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row and row[1] > time.time():
            return json.loads(row[0])
    except Exception:
        pass
    return None


def _cache_set_recommend(key: str, value, ttl: int):
    """Store recommendation result in cache."""
    import sqlite3
    try:
        conn = sqlite3.connect(str(CACHE_DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), time.time() + ttl),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


@router.get("/recommend", response_model=RecommendResponse)
async def get_recommendations(
    username: str = Query(..., description="Bangumi username"),
    subject_type: int = Query(2, description="Subject type: 1=book, 2=anime, 3=music, 4=game, 6=real"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=50),
    nsfw: bool = Query(False, description="Include NSFW content"),
):
    """Get personalized recommendations for a Bangumi user."""
    if subject_type not in SUBJECT_TYPES:
        raise HTTPException(400, f"Invalid subject_type. Valid: {list(SUBJECT_TYPES.keys())}")

    if not deps.bgm_client:
        raise HTTPException(503, "Service not initialized")

    # Check cache
    cache_key = f"recommend:{username}:{subject_type}:{limit}:{nsfw}"
    cached = _cache_get_recommend(cache_key)
    if cached:
        return cached

    # Fetch user collections (all types for cross-type understanding)
    try:
        collections = await deps.bgm_client.get_user_collections(username, subject_type=subject_type)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch user collections: {e}")

    if not collections:
        raise HTTPException(404, f"User '{username}' not found or has no collections for this type")

    parsed = parse_api_collections(collections)

    # Also fetch anime collections if requesting non-anime (for cross-type taste)
    all_collections = parsed
    if subject_type != 2:
        try:
            anime_cols = await deps.bgm_client.get_user_collections(username, subject_type=2)
            all_collections = parse_api_collections(anime_cols) + parsed
        except Exception:
            pass

    # CF predictions (anime only)
    cf_scores = []
    cf_available = False
    if subject_type == 2 and deps.cf_recommender and deps.cf_recommender.loaded:
        cf_scores = deps.cf_recommender.predict(parsed, top_n=limit * 5)
        cf_available = True

    # Content-based predictions
    content_scores = []
    if deps.content_recommender and deps.content_recommender.loaded:
        collected_ids = {c["subject_id"] for c in parsed}
        content_scores = deps.content_recommender.recommend(
            all_collections,
            subject_type=subject_type,
            top_n=limit * 5,
            exclude_ids=collected_ids,
        )

    # Hybrid ranking
    if deps.hybrid_ranker:
        ranked = deps.hybrid_ranker.rank(
            cf_scores=cf_scores,
            content_scores=content_scores,
            user_collections=parsed,
            subject_type=subject_type,
            limit=limit,
            filter_nsfw=not nsfw,
            content_recommender=deps.content_recommender,
        )
    else:
        # Fallback: just use content scores
        ranked = [
            {
                "subject_id": sid,
                "name_cn": "",
                "name": "",
                "score": score,
                "reason": "",
                "bangumi_score": None,
                "tags": [],
                "image_url": f"https://api.bgm.tv/v0/subjects/{sid}/image?type=medium",
                "subject_type": subject_type,
            }
            for sid, score in content_scores[:limit]
        ]

    result = {
        "username": username,
        "subject_type": subject_type,
        "recommendations": ranked,
        "total_collections": len(parsed),
        "cf_available": cf_available,
    }

    # Cache result
    _cache_set_recommend(cache_key, result, CACHE_TTL_RECOMMEND_RESULT)

    return result


@router.get("/user/{username}/profile", response_model=UserProfile)
async def get_user_profile(username: str):
    """Get user collection overview."""
    if not deps.bgm_client:
        raise HTTPException(503, "Service not initialized")

    all_collections = []
    for st in SUBJECT_TYPES:
        try:
            cols = await deps.bgm_client.get_user_collections(username, subject_type=st)
            for c in cols:
                c["subject_type_override"] = st
            all_collections.extend(parse_api_collections(cols))
        except Exception:
            continue

    if not all_collections:
        raise HTTPException(404, f"User '{username}' not found or has no collections")

    stats = get_collection_stats(all_collections)
    return UserProfile(username=username, **stats)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check."""
    return HealthResponse(
        status="ok",
        cf_model_loaded=bool(deps.cf_recommender and deps.cf_recommender.loaded),
        faiss_indices_loaded=list(deps.content_recommender.indices.keys()) if deps.content_recommender else [],
        total_subjects=len(deps.hybrid_ranker.subjects_meta) if deps.hybrid_ranker and deps.hybrid_ranker.subjects_meta is not None else 0,
    )
