/**
 * The CodeFrog session: one JWT in sessionStorage. It is the ONLY thing this app ever puts in
 * browser storage. Never store the GitHub access token (the backend keeps it, encrypted), a
 * client secret, repository contents, or anything else here, and never use persistent (local) storage.
 *
 * sessionStorage is cleared when the tab closes, and is not shared with other tabs.
 */

export const SESSION_KEY = "codefrog.access_token";

const MAX_TOKEN_LENGTH = 4096;
/** A JWT: base64url segments separated by dots (padding tolerated). */
const TOKEN_PATTERN = /^[A-Za-z0-9._~+/-]+={0,2}$/;

/** The part of the Web Storage API used here, so tests can supply a fake. */
export interface TokenStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function browserStorage(): TokenStorage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null; // storage can be blocked (privacy settings, sandboxed frames)
  }
}

export function isValidToken(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= MAX_TOKEN_LENGTH && TOKEN_PATTERN.test(value);
}

/** The stored token, or null if there is none, it is malformed, or storage is unavailable. */
export function getSessionToken(storage: TokenStorage | null = browserStorage()): string | null {
  try {
    const token = storage?.getItem(SESSION_KEY) ?? null;
    return isValidToken(token) ? token : null;
  } catch {
    return null;
  }
}

/** Store a token. Returns false (and stores nothing) if it is not a plausible token or storage fails. */
export function setSessionToken(token: string, storage: TokenStorage | null = browserStorage()): boolean {
  if (!isValidToken(token) || storage === null) return false;
  try {
    storage.setItem(SESSION_KEY, token);
    return true;
  } catch {
    return false;
  }
}

export function clearSession(storage: TokenStorage | null = browserStorage()): void {
  try {
    storage?.removeItem(SESSION_KEY);
  } catch {
    // nothing to clear
  }
}

export function hasSession(storage: TokenStorage | null = browserStorage()): boolean {
  return getSessionToken(storage) !== null;
}
