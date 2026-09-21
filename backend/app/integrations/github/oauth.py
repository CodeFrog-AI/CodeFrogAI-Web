"""Small, secret-safe GitHub OAuth HTTP client."""

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from app.core.config import get_settings


AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
USER_URL = "https://api.github.com/user"
USER_EMAILS_URL = "https://api.github.com/user/emails"
# The identity flow needs these regardless of what is configured.
REQUIRED_IDENTITY_SCOPES = ("read:user", "user:email")


class GitHubOAuthError(RuntimeError):
    """Raised for provider failures without preserving provider response content."""


@dataclass(frozen=True)
class GitHubIdentity:
    """The minimum verified GitHub identity required for account linking."""

    github_user_id: int
    login: str
    email: str
    name: str | None


def oauth_scopes() -> str:
    """The configured scopes (GITHUB_OAUTH_SCOPES), always including the identity scopes."""

    configured = get_settings().github_oauth_scopes.split()
    extra = [scope for scope in configured if scope not in REQUIRED_IDENTITY_SCOPES]
    return " ".join([*REQUIRED_IDENTITY_SCOPES, *extra])


class GitHubOAuthClient:
    """Perform only the OAuth operations needed for GitHub authentication."""

    def build_authorization_url(self, state: str) -> str:
        """Build the trusted GitHub authorization URL with the configured scopes."""

        settings = get_settings()
        query = urlencode(
            {
                "client_id": settings.github_client_id,
                "redirect_uri": str(settings.github_redirect_uri),
                "scope": oauth_scopes(),
                "state": state,
            }
        )
        return f"{AUTHORIZE_URL}?{query}"

    def exchange_code(self, code: str) -> str:
        """Exchange an authorization code for a short-lived, in-memory provider token."""

        settings = get_settings()
        try:
            response = httpx.post(
                ACCESS_TOKEN_URL,
                data={
                    "client_id": settings.github_client_id,
                    "client_secret": settings.github_client_secret.get_secret_value(),
                    "code": code,
                    "redirect_uri": str(settings.github_redirect_uri),
                },
                headers={"Accept": "application/json"},
                timeout=10.0,
            )
            response.raise_for_status()
            access_token = response.json().get("access_token")
        except (httpx.HTTPError, ValueError, AttributeError):
            raise GitHubOAuthError("GitHub token exchange failed") from None
        if not isinstance(access_token, str) or not access_token:
            raise GitHubOAuthError("GitHub token exchange failed")
        return access_token

    def get_identity(self, access_token: str) -> GitHubIdentity:
        """Retrieve profile data and a verified email for safe local-account linking."""

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {access_token}",
        }
        try:
            profile_response = httpx.get(USER_URL, headers=headers, timeout=10.0)
            profile_response.raise_for_status()
            profile = profile_response.json()
            emails_response = httpx.get(USER_EMAILS_URL, headers=headers, timeout=10.0)
            emails_response.raise_for_status()
            verified_email = self._verified_email(emails_response.json())
            github_user_id = profile.get("id")
            login = profile.get("login")
        except (httpx.HTTPError, ValueError, AttributeError):
            raise GitHubOAuthError("GitHub profile retrieval failed") from None

        if not isinstance(github_user_id, int) or not isinstance(login, str) or not verified_email:
            raise GitHubOAuthError("GitHub profile retrieval failed")
        name = profile.get("name")
        return GitHubIdentity(
            github_user_id=github_user_id,
            login=login,
            email=verified_email,
            name=name if isinstance(name, str) else None,
        )

    @staticmethod
    def _verified_email(emails: Any) -> str | None:
        """Prefer the primary verified address, accepting no unverified address."""

        if not isinstance(emails, list):
            return None
        verified = [
            item["email"].strip().lower()
            for item in emails
            if isinstance(item, dict)
            and item.get("verified") is True
            and isinstance(item.get("email"), str)
        ]
        primary = next(
            (
                item["email"].strip().lower()
                for item in emails
                if isinstance(item, dict)
                and item.get("verified") is True
                and item.get("primary") is True
                and isinstance(item.get("email"), str)
            ),
            None,
        )
        return primary or (verified[0] if verified else None)
