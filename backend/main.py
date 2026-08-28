from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.routes.health import build_health_response
from app.core.exception_handlers import register_exception_handlers
from app.db.database import check_database_connection

app = FastAPI(
    title="CodeFrog AI API",
    description="AI-powered Software Engineer API",
    version="0.1.0",
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
)

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
