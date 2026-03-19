"""Application configuration."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"

# Bangumi API
BGM_API_BASE = "https://api.bgm.tv/v0"
BGM_API_RATE_LIMIT = 3  # requests per second

# Cache settings (SQLite)
CACHE_DB_PATH = DATA_DIR / "cache.db"
CACHE_TTL_USER_COLLECTION = 3600       # 1 hour
CACHE_TTL_RECOMMEND_RESULT = 21600     # 6 hours
CACHE_TTL_SUBJECT_DETAIL = 86400       # 24 hours

# Recommendation settings
DEFAULT_LIMIT = 20
CF_WEIGHT_DEFAULT = 0.7       # α for hybrid: CF weight
CF_WEIGHT_COLD_START = 0.4    # α when user has <10 anime collections
COLD_START_THRESHOLD = 10     # Min anime collections for full CF weight
DIVERSITY_LAMBDA = 0.3        # MMR diversity parameter
POPULARITY_WEIGHT = 0.05      # Micro popularity boost weight
NSFW_DEFAULT = False          # Filter NSFW by default

# Subject types
SUBJECT_TYPES = {1: "书籍", 2: "动画", 3: "音乐", 4: "游戏", 6: "三次元"}
