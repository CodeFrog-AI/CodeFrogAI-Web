"""GitHub OAuth routes that hand the browser a CodeFrog JWT, never a provider token.

The callback always ends by redirecting to the web frontend (FRONTEND_URL):

    success:  <FRONTEND_URL>/auth/callback#access_token=<CodeFrog JWT>
    failure:  <FRONTEND_URL>/auth/callback#error=<fixed error code>

The JWT travels in the URL *fragment*, which browsers never send to any server, and never in
a query string. Error codes are a fixed, safe vocabulary: no provider text, tokens, stack
traces, or database errors are ever put in the redirect.
"""

import hmac
import logging
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Cookie, Depends, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth.github_service import resolve_github_identity, store_github_token
from app.auth.oauth_state import oauth_state_store
from app.auth.security import create_access_token
from app.core.config import get_settings
from app.core.exceptions import ApplicationError
from app.db.database import get_db
from app.integrations.github.oauth import GitHubOAuthClient, GitHubOAuthError
from app.integrations.github.tokens import TokenEncryptionError, require_token_encryption


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth/github", tags=["authentication"])
STATE_COOKIE = "github_oauth_state"
STATE_TTL_SECONDS = 600
CALLBACK_COOKIE_PATH = "/api/v1/auth/github/callback"

# The complete set of error codes the frontend can receive.
ERROR_INVALID_STATE = "invalid_state"
ERROR_ACCESS_DENIED = "access_denied"
ERROR_AUTHORIZATION_FAILED = "authorization_failed"
ERROR_ACCOUNT_UNAVAILABLE = "account_unavailable"
ERROR_SERVER = "server_error"


def _frontend_redirect(fragment: dict[str, str], *, clear_state_cookie: bool = True) -> RedirectResponse:
    """Redirect to the frontend's callback page with `fragment` (already safe values) after the `#`."""

    target = f"{get_settings().frontend_url}/auth/callback#{urlencode(fragment)}"
    response = RedirectResponse(target, status_code=302)
    # The fragment may hold a JWT: keep it out of caches and Referer headers.
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    if clear_state_cookie:
        response.delete_cookie(STATE_COOKIE, path=CALLBACK_COOKIE_PATH)
    return response


def _failure(code: str) -> RedirectResponse:
    logger.warning("GitHub OAuth callback failed reason=%s", code)
    return _frontend_redirect({"error": code})


@router.get("/login")
def github_login() -> RedirectResponse:
    """Start GitHub OAuth with an unpredictable, browser-bound CSRF state."""

    try:
        require_token_encryption()  # do not send the user to GitHub if we could not keep the result
    except TokenEncryptionError:
        logger.error("GitHub OAuth cannot start: token encryption is not configured")
        return _frontend_redirect({"error": ERROR_SERVER}, clear_state_cookie=False)

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
        path=CALLBACK_COOKIE_PATH,
    )
    logger.info("GitHub OAuth login initiated")
    return response


@router.get("/callback")
def github_callback(
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
    browser_state: Annotated[str | None, Cookie(alias=STATE_COOKIE)] = None,
    session: Session = Depends(get_db),
) -> RedirectResponse:
    """Validate OAuth state, resolve a GitHub identity, and redirect to the frontend with a CodeFrog JWT."""

    # CSRF protection comes first and is unchanged: the state must match the browser's cookie and
    # be one we issued, and it is single-use.
    if state is None or browser_state is None or not hmac.compare_digest(state, browser_state):
        return _failure(ERROR_INVALID_STATE)
    if not oauth_state_store.consume(state):
        return _failure(ERROR_INVALID_STATE)

    if error is not None:  # the user declined, or GitHub refused; nothing else to do
        return _failure(ERROR_ACCESS_DENIED if error == "access_denied" else ERROR_AUTHORIZATION_FAILED)
    if not code:
        return _failure(ERROR_AUTHORIZATION_FAILED)

    try:
        require_token_encryption()  # fail before doing any work if the token could not be stored
    except TokenEncryptionError:
        logger.error("GitHub OAuth cannot complete: token encryption is not configured")
        return _failure(ERROR_SERVER)

    try:
        client = GitHubOAuthClient()
        github_access_token = client.exchange_code(code)
        identity = client.get_identity(github_access_token)
    except GitHubOAuthError:
        return _failure(ERROR_AUTHORIZATION_FAILED)

    try:
        user = resolve_github_identity(session, identity)
        if user.status != "active":
            return _failure(ERROR_ACCOUNT_UNAVAILABLE)
        store_github_token(session, identity.github_user_id, github_access_token)
    except TokenEncryptionError:
        return _failure(ERROR_SERVER)
    except ApplicationError:
        return _failure(ERROR_AUTHORIZATION_FAILED)
    except SQLAlchemyError:
        session.rollback()
        logger.error("GitHub OAuth callback failed with a database error")
        return _failure(ERROR_SERVER)

    logger.info("GitHub OAuth identity resolved")
    return _frontend_redirect({"access_token": create_access_token(user.id)})
