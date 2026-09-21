"""Shared backend test configuration."""

import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet


# Settings are instantiated while application modules are imported. This URL is
# never contacted by unit tests because their database boundary is mocked.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://codefrog:codefrog_dev@localhost:5432/codefrog"
)
os.environ.setdefault("AUTH_SECRET_KEY", "test-only-secret-key-that-is-at-least-32-characters")
os.environ.setdefault("GITHUB_CLIENT_ID", "test-github-client-id")
os.environ.setdefault("GITHUB_CLIENT_SECRET", "test-github-client-secret")
os.environ.setdefault(
    "GITHUB_REDIRECT_URI", "http://localhost:8000/api/v1/auth/github/callback"
)

os.environ.setdefault("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
# Pinned so a developer's real .env cannot change what the OAuth tests see.
os.environ.setdefault("GITHUB_OAUTH_SCOPES", "read:user user:email repo")
os.environ.setdefault("FRONTEND_URL", "http://localhost:3000")

# Permit `pytest backend/tests` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
