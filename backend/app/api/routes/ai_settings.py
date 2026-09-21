"""The signed-in user's AI provider settings. API keys go in, never out."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.ai_settings.service import build_settings_view, remove_api_key, update_ai_settings
from app.auth.dependencies import get_current_user
from app.db.database import get_db
from app.db.models import User
from app.schemas.ai_settings import AISettingsResponse, AISettingsUpdate

router = APIRouter(prefix="/settings/ai", tags=["settings"])

CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[Session, Depends(get_db)]


@router.get("", response_model=AISettingsResponse)
def get_ai_settings(current_user: CurrentUser, session: DbSession) -> AISettingsResponse:
    """Whether each key is configured (with its last 4 characters), the models, and whose key is used."""

    return build_settings_view(session, current_user)


@router.put("", response_model=AISettingsResponse)
def update_settings(payload: AISettingsUpdate, current_user: CurrentUser, session: DbSession) -> AISettingsResponse:
    """A partial update of the LLM and/or embedding settings. Keys are encrypted and never returned."""

    return update_ai_settings(session, current_user, payload)


@router.delete("/llm-key", response_model=AISettingsResponse)
def delete_llm_key(current_user: CurrentUser, session: DbSession) -> AISettingsResponse:
    return remove_api_key(session, current_user, "llm")


@router.delete("/embedding-key", response_model=AISettingsResponse)
def delete_embedding_key(current_user: CurrentUser, session: DbSession) -> AISettingsResponse:
    return remove_api_key(session, current_user, "embedding")
