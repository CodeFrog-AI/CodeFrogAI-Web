"""Top-level router composition for the public API."""

from fastapi import APIRouter

from app.api.routes import auth, health, repositories, tasks, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(repositories.router)
api_router.include_router(tasks.router)
