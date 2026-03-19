"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import deps
from backend.routers import recommend


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indices at startup, cleanup at shutdown."""
    deps.init_all()
    yield
    await deps.cleanup()


app = FastAPI(
    title="Bangumi Recommender",
    description="Personalized anime/book/music/game recommendations based on your Bangumi collections",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(recommend.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
