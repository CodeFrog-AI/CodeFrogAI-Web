from fastapi import FastAPI

app = FastAPI(
    title="CodeFrog AI API",
    description="AI-powered Software Engineer API",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Welcome to CodeFrog AI 🐸",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }