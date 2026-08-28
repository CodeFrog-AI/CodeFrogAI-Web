"""Top-level router composition for the public API."""

from fastapi import APIRouter

from app.api.routes import health, repositories, tasks, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(users.router)
api_router.include_router(repositories.router)
api_router.include_router(tasks.router)
