"""Shared backend test configuration."""

import os
import sys
from pathlib import Path


# Settings are instantiated while application modules are imported. This URL is
# never contacted by unit tests because their database boundary is mocked.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://codefrog:codefrog_dev@localhost:5432/codefrog"
)

# Permit `pytest backend/tests` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
