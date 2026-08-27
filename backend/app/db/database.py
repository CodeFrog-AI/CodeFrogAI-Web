"""SQLAlchemy engine, request sessions, and connectivity checks."""

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


class DatabaseConnectionError(RuntimeError):
    """Raised when PostgreSQL cannot be reached safely."""


def create_database_engine() -> Engine:
    """Create the PostgreSQL engine from the environment-provided URL."""

    return create_engine(str(get_settings().database_url), pool_pre_ping=True)


engine = create_database_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and always close it after the request."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_database_connection() -> None:
    """Verify that PostgreSQL accepts a lightweight query without leaking URLs."""

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as error:
        raise DatabaseConnectionError("Database connection failed") from error
