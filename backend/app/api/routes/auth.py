"""Local registration, login, and protected identity endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token
from app.auth.service import authenticate_user, register_user
from app.core.exceptions import UnauthorizedError
from app.db.database import get_db
from app.db.models import User
from app.schemas.auth import AccessTokenResponse, LoginRequest, PublicUserResponse, RegistrationRequest


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=PublicUserResponse, status_code=201)
def register(
    payload: RegistrationRequest,
    session: Annotated[Session, Depends(get_db)],
) -> User:
    """Register a user while storing only an Argon2 password hash."""

    return register_user(
        session, email=payload.email, name=payload.name, password=payload.password
    )


@router.post("/login", response_model=AccessTokenResponse)
def login(
    payload: LoginRequest,
    session: Annotated[Session, Depends(get_db)],
) -> AccessTokenResponse:
    """Verify local credentials and issue a short-lived bearer access token."""

    user = authenticate_user(session, email=payload.email, password=payload.password)
    if user is None:
        raise UnauthorizedError("Invalid email or password")
    return AccessTokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=PublicUserResponse)
def get_me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Return the authenticated user's public profile."""

    return current_user
