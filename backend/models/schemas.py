"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    subject_id: int
    name_cn: str = ""
    name: str = ""
    score: float = Field(description="Recommendation confidence score")
    reason: str = ""
    bangumi_score: float | None = None
    tags: list[str] = []
    image_url: str = ""
    subject_type: int = 2


class RecommendResponse(BaseModel):
    username: str
    subject_type: int
    recommendations: list[RecommendationItem]
    total_collections: int = 0
    cf_available: bool = True


class UserProfile(BaseModel):
    username: str
    anime_count: int = 0
    book_count: int = 0
    music_count: int = 0
    game_count: int = 0
    real_count: int = 0
    top_tags: list[dict] = []
    avg_rating: float | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    cf_model_loaded: bool = False
    faiss_indices_loaded: list[int] = []
    total_subjects: int = 0
