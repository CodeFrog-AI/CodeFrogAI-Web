import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.routes.health import build_health_response
from app.core.config import get_settings
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.db.database import check_database_connection

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("codefrog.application")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Record concise application lifecycle events."""

    logger.info("Application starting")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title=settings.app_name,
    description="AI-powered Software Engineer API",
    version="0.1.0",
    lifespan=lifespan,
)

register_exception_handlers(app)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to CodeFrog AI 🐸",
        "status": "running",
    }


@app.get("/health")
def health():
    """Legacy health endpoint retained for existing clients."""

    return build_health_response(check_database_connection)
