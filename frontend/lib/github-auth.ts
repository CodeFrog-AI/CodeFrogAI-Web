/**
 * The web GitHub sign-in flow:
 *
 *   Connect GitHub -> (full-page navigation) backend /auth/github/login -> GitHub -> backend
 *   callback -> redirect to <frontend>/auth/callback#access_token=<CodeFrog JWT>
 *
 * The callback page reads the token from the URL fragment, removes it from the address bar and
 * history at once, keeps it in session storage (via session.ts), and confirms it with GET /auth/me.
 * The token is never rendered, logged, or sent anywhere except the Authorization header.
 */

import { ApiError, getCurrentUser, githubLoginUrl, type CurrentUser } from "@/lib/api";
import { isValidToken, setSessionToken, type TokenStorage } from "@/lib/session";

export type AuthErrorCode =
  | "invalid_state"
  | "access_denied"
  | "authorization_failed"
  | "account_unavailable"
  | "server_error"
  | "storage_unavailable"
  | "session_rejected"
  | "backend_unavailable"
  | "no_response";

/** Fixed messages: the fragment's error text is never shown, only mapped through this table. */
export const AUTH_ERROR_MESSAGES: Record<AuthErrorCode, string> = {
  invalid_state: "The sign-in request could not be verified (it may have expired). Please try again.",
  access_denied: "GitHub authorization was cancelled. You can try again whenever you are ready.",
  authorization_failed: "GitHub sign-in could not be completed. Please try again.",
  account_unavailable: "This account cannot sign in right now.",
  server_error: "CodeFrog could not finish connecting GitHub. Please try again, or contact the administrator if it keeps happening.",
  storage_unavailable: "Your browser blocked storing the session, so sign-in cannot continue. Allow site storage and try again.",
  session_rejected: "CodeFrog did not accept the new session. Please try again.",
  backend_unavailable: "Could not reach the CodeFrog backend. Check that it is running and try again.",
  no_response: "No sign-in information was received. Start again with Connect GitHub.",
};

const CODES = Object.keys(AUTH_ERROR_MESSAGES) as AuthErrorCode[];

export type CallbackOutcome = { kind: "token"; token: string } | { kind: "error"; code: AuthErrorCode } | { kind: "empty" };

/** Read `#access_token=...` or `#error=...` from a URL hash. Anything else is empty or a generic failure. */
export function parseCallbackHash(hash: string): CallbackOutcome {
  const params = new URLSearchParams(hash.startsWith("#") ? hash.slice(1) : hash);
  const token = params.get("access_token");
  if (token !== null) {
    return isValidToken(token) ? { kind: "token", token } : { kind: "error", code: "authorization_failed" };
  }
  const error = params.get("error");
  if (error !== null) {
    const known = (CODES as string[]).includes(error) && error !== "storage_unavailable";
    return { kind: "error", code: known ? (error as AuthErrorCode) : "authorization_failed" };
  }
  return { kind: "empty" };
}

/** Start GitHub OAuth. A normal full-page navigation: never fetch, an iframe, or a popup. */
export function connectGitHub(navigate: (url: string) => void = (url) => window.location.assign(url)): void {
  navigate(githubLoginUrl());
}

export interface SignInDeps {
  storage?: TokenStorage | null;
  /** Replace the address-bar URL with one that has no fragment. */
  replaceUrl: () => void;
  getCurrentUser?: () => Promise<CurrentUser>;
}

export type SignInResult = { status: "signed-in"; user: CurrentUser } | { status: "failed"; code: AuthErrorCode };

/**
 * Finish sign-in from the callback URL's hash. The URL is cleaned first, before anything else can
 * fail or be slow, so the JWT never lingers in the address bar or history.
 */
export async function completeSignIn(hash: string, deps: SignInDeps): Promise<SignInResult> {
  const outcome = parseCallbackHash(hash);
  deps.replaceUrl();

  if (outcome.kind === "empty") return { status: "failed", code: "no_response" };
  if (outcome.kind === "error") return { status: "failed", code: outcome.code };

  const stored = deps.storage === undefined ? setSessionToken(outcome.token) : setSessionToken(outcome.token, deps.storage);
  if (!stored) return { status: "failed", code: "storage_unavailable" };

  try {
    const user = await (deps.getCurrentUser ?? getCurrentUser)();
    return { status: "signed-in", user };
  } catch (error) {
    if (error instanceof ApiError && error.code === "NETWORK") return { status: "failed", code: "backend_unavailable" };
    if (error instanceof ApiError && (error.code === "UNAUTHORIZED" || error.code === "FORBIDDEN")) {
      return { status: "failed", code: "session_rejected" };
    }
    return { status: "failed", code: "authorization_failed" };
  }
}
