"""Short-lived, single-use OAuth state storage for CSRF protection."""

import hashlib
import secrets
import threading
import time


class OAuthStateStore:
    """Keep only state digests in process memory until they are consumed or expire."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self.ttl_seconds = ttl_seconds
        self._states: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        """Generate an unpredictable state value and retain only its digest."""

        state = secrets.token_urlsafe(32)
        with self._lock:
            self._purge_expired()
            self._states[self._digest(state)] = time.monotonic() + self.ttl_seconds
        return state

    def consume(self, state: str) -> bool:
        """Validate and delete state, making every valid state single-use."""

        with self._lock:
            self._purge_expired()
            return self._states.pop(self._digest(state), None) is not None

    @staticmethod
    def _digest(state: str) -> str:
        return hashlib.sha256(state.encode("utf-8")).hexdigest()

    def _purge_expired(self) -> None:
        now = time.monotonic()
        self._states = {
            digest: expires_at
            for digest, expires_at in self._states.items()
            if expires_at > now
        }


oauth_state_store = OAuthStateStore()
