"""Top-level router composition for the public API."""

from fastapi import APIRouter

from app.api.routes import auth, github_oauth, health, repositories, repository_git, repository_pr, repository_pr_fix, tasks, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(github_oauth.router)
api_router.include_router(users.router)
api_router.include_router(repositories.router)
api_router.include_router(repository_git.router)
api_router.include_router(repository_pr.router)
api_router.include_router(repository_pr_fix.router)
api_router.include_router(tasks.router)
