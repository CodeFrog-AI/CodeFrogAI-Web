"""Reusable FastAPI dependencies for authenticated routes."""

from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.security import decode_access_token
from app.auth.service import find_user_by_id
from app.core.exceptions import UnauthorizedError
from app.db.database import get_db
from app.db.models import User


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    session: Annotated[Session, Depends(get_db)],
) -> User:
    """Resolve an active user from a valid bearer token or raise a safe 401."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthorizedError()

    user_id = decode_access_token(credentials.credentials)
    if user_id is None:
        raise UnauthorizedError()

    user = find_user_by_id(session, user_id)
    if user is None or user.status != "active":
        raise UnauthorizedError()
    return user
