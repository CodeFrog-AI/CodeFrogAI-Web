"""Database operations for local user authentication."""

from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.security import hash_password, verify_password
from app.core.exceptions import ConflictError
from app.db.models import User


def find_user_by_email(session: Session, email: str) -> User | None:
    """Find a user by its normalized, unique email address."""

    return session.query(User).filter(User.email == email).first()


def register_user(session: Session, *, email: str, name: str | None, password: str) -> User:
    """Create a local user while persisting only the Argon2 password hash."""

    if find_user_by_email(session, email) is not None:
        raise ConflictError("An account with this email already exists")

    user = User(
        email=email,
        name=name,
        status="active",
        password_hash=hash_password(password),
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise ConflictError("An account with this email already exists") from None
    session.refresh(user)
    return user


def authenticate_user(session: Session, *, email: str, password: str) -> User | None:
    """Return an active user only when its local password verifies."""

    user = find_user_by_email(session, email)
    if user is None or user.status != "active" or user.password_hash is None:
        return None
    return user if verify_password(password, user.password_hash) else None


def find_user_by_id(session: Session, user_id: UUID) -> User | None:
    """Load a user for a previously verified token subject."""

    return session.get(User, user_id)
