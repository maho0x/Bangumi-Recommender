"""Singleton dependencies: load models, indices, and metadata once at startup."""

from backend.services.bgm_api import BangumiAPIClient
from backend.services.cf_recommender import CFRecommender
from backend.services.content_recommender import ContentRecommender
from backend.services.hybrid_ranker import HybridRanker

# Singletons
bgm_client: BangumiAPIClient | None = None
cf_recommender: CFRecommender | None = None
content_recommender: ContentRecommender | None = None
hybrid_ranker: HybridRanker | None = None


def init_all():
    """Initialize all services. Called at FastAPI startup."""
    global bgm_client, cf_recommender, content_recommender, hybrid_ranker

    print("Initializing services ...")

    bgm_client = BangumiAPIClient()

    cf_recommender = CFRecommender()
    cf_recommender.load()

    content_recommender = ContentRecommender()
    content_recommender.load()

    hybrid_ranker = HybridRanker()
    hybrid_ranker.load()

    print("All services initialized.")


async def cleanup():
    """Cleanup on shutdown."""
    global bgm_client
    if bgm_client:
        await bgm_client.close()
