"""Safe request and response schemas for local authentication."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CredentialsRequest(BaseModel):
    """Shared local-password credentials; passwords require at least 12 characters."""

    email: str = Field(max_length=320)
    password: str = Field(min_length=12, max_length=1024)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
            raise ValueError("A valid email address is required")
        return normalized


class RegistrationRequest(CredentialsRequest):
    """Registration payload with an optional display name."""

    name: str | None = Field(default=None, max_length=255)


class LoginRequest(CredentialsRequest):
    """Local login payload."""


class PublicUserResponse(BaseModel):
    """The only user shape exposed publicly; no security fields are present."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    name: str | None
    status: str
    created_at: datetime


class AccessTokenResponse(BaseModel):
    """Bearer access token returned only after successful credential verification."""

    access_token: str
    token_type: str = "bearer"
