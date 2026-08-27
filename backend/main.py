from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import DatabaseConnectionError, check_database_connection

app = FastAPI(
    title="CodeFrog AI API",
    description="AI-powered Software Engineer API",
    version="0.1.0",
)


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


@app.get("/")
def root():
    return {
        "message": "Welcome to CodeFrog AI 🐸",
        "status": "running",
    }


@app.get("/health")
def health():
    try:
        check_database_connection()
    except DatabaseConnectionError as error:
        raise HTTPException(status_code=503, detail="Database is unavailable") from error

    return {
        "status": "healthy",
        "database": "healthy",
    }
