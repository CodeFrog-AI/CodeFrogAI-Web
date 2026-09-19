"""GitHub OAuth routes that issue CodeFrog JWTs, never provider tokens."""

import hmac
import logging
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Query, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.auth.github_service import resolve_github_identity, store_github_token
from app.auth.oauth_state import oauth_state_store
from app.auth.security import create_access_token
from app.core.config import get_settings
from app.core.exceptions import BadRequestError, UnauthorizedError
from app.db.database import get_db
from app.integrations.github.oauth import GitHubOAuthClient, GitHubOAuthError
from app.schemas.auth import AccessTokenResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth/github", tags=["authentication"])
STATE_COOKIE = "github_oauth_state"
STATE_TTL_SECONDS = 600


@router.get("/login")
def github_login() -> RedirectResponse:
    """Start GitHub OAuth with an unpredictable, browser-bound CSRF state."""

    state = oauth_state_store.create()
    authorization_url = GitHubOAuthClient().build_authorization_url(state)
    settings = get_settings()
    response = RedirectResponse(authorization_url, status_code=302)
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=STATE_TTL_SECONDS,
        httponly=True,
        secure=settings.app_env == "production",
        samesite="lax",
        path="/api/v1/auth/github/callback",
    )
    logger.info("GitHub OAuth login initiated")
    return response


@router.get("/callback", response_model=AccessTokenResponse)
def github_callback(
    response: Response,
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    browser_state: Annotated[str | None, Cookie(alias=STATE_COOKIE)] = None,
    session: Session = Depends(get_db),
) -> AccessTokenResponse:
    """Validate OAuth state, resolve a GitHub identity, and return a CodeFrog JWT."""

    if state is None or browser_state is None or not hmac.compare_digest(state, browser_state):
        raise UnauthorizedError("GitHub OAuth state is invalid")
    if not oauth_state_store.consume(state):
        raise UnauthorizedError("GitHub OAuth state is invalid")
    response.delete_cookie(STATE_COOKIE, path="/api/v1/auth/github/callback")
    if not code:
        raise BadRequestError("GitHub OAuth authorization could not be completed")

    try:
        client = GitHubOAuthClient()
        github_access_token = client.exchange_code(code)
        identity = client.get_identity(github_access_token)
    except GitHubOAuthError:
        raise BadRequestError("GitHub OAuth authorization could not be completed") from None

    user = resolve_github_identity(session, identity)
    if user.status != "active":
        raise UnauthorizedError("GitHub authentication could not be completed")
    store_github_token(session, identity.github_user_id, github_access_token)

    logger.info("GitHub OAuth identity resolved")
    return AccessTokenResponse(access_token=create_access_token(user.id))
