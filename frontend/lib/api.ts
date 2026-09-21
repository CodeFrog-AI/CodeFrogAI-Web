/**
 * The CodeFrog backend client. Every request goes to the one configured backend
 * (NEXT_PUBLIC_API_URL), only to `/api/v1/...` paths, with the session JWT as a Bearer header.
 *
 * Failures become typed `ApiError`s with fixed, user-safe messages: the response body (which
 * could hold a stack trace or provider text) is never read for errors or shown.
 *  - 401: the session is cleared, so the UI can send the user back to Connect GitHub.
 *  - 403: surfaced as FORBIDDEN, so the UI can offer to reconnect.
 */

import { clearSession, getSessionToken } from "@/lib/session";

export type ApiErrorCode =
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "NOT_FOUND"
  | "CONFLICT"
  | "BAD_REQUEST"
  | "SERVER_ERROR"
  | "NETWORK"
  | "INVALID_RESPONSE"
  | "CONFIG";

export const API_ERROR_MESSAGES: Record<ApiErrorCode, string> = {
  UNAUTHORIZED: "Your session has expired. Connect GitHub to continue.",
  FORBIDDEN: "GitHub or CodeFrog denied access. Reconnect GitHub and try again.",
  NOT_FOUND: "That could not be found.",
  CONFLICT: "That conflicts with the current state (for example, it is already connected elsewhere).",
  BAD_REQUEST: "The request was not accepted.",
  SERVER_ERROR: "CodeFrog or GitHub had a problem. Please try again in a moment.",
  NETWORK: "Could not reach the CodeFrog backend. Check that it is running and try again.",
  INVALID_RESPONSE: "The backend returned something CodeFrog could not understand.",
  CONFIG: "The CodeFrog backend address is not configured correctly.",
};

export class ApiError extends Error {
  readonly code: ApiErrorCode;
  readonly status: number | null;

  constructor(code: ApiErrorCode, status: number | null = null) {
    super(API_ERROR_MESSAGES[code]);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

/** For local development only; set NEXT_PUBLIC_API_URL for any other environment. */
export const DEFAULT_API_URL = "http://localhost:8000";
export const GITHUB_LOGIN_PATH = "/api/v1/auth/github/login";

/** The backend origin: http(s), a host, no credentials, path, query, or fragment. */
export function normalizeBaseUrl(raw: string | undefined): string {
  const value = (raw ?? "").trim() || DEFAULT_API_URL;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new ApiError("CONFIG");
  }
  const bare = url.pathname === "/" || url.pathname === "";
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password || !bare || url.search || url.hash) {
    throw new ApiError("CONFIG");
  }
  return url.origin;
}

export function getApiBaseUrl(): string {
  // The variable must be referenced literally so Next.js can inline it at build time.
  return normalizeBaseUrl(process.env.NEXT_PUBLIC_API_URL);
}

/** The full-page navigation target that starts GitHub OAuth. */
export function githubLoginUrl(baseUrl: string = getApiBaseUrl()): string {
  return `${baseUrl}${GITHUB_LOGIN_PATH}`;
}

export interface ApiDeps {
  fetch: (input: string, init: RequestInit) => Promise<Response>;
  baseUrl: () => string;
  getToken: () => string | null;
  clearSession: () => void;
}

function defaultDeps(): ApiDeps {
  return {
    fetch: (input, init) => fetch(input, init),
    baseUrl: getApiBaseUrl,
    getToken: () => getSessionToken(),
    clearSession: () => clearSession(),
  };
}

const API_PATH = /^\/api\/v1\/[A-Za-z0-9_\-/]*$/;

export interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
}

/**
 * Make one JSON request. `parse` validates the response and throws if it is not what was
 * expected; anything unexpected becomes INVALID_RESPONSE.
 */
export async function apiRequest<T>(
  path: string,
  parse: (value: unknown) => T,
  options: RequestOptions = {},
  deps: ApiDeps = defaultDeps(),
): Promise<T> {
  // Only fixed backend API paths: this client cannot be pointed at another host or path.
  if (!API_PATH.test(path) || path.includes("//") || path.includes("..")) throw new ApiError("CONFIG");

  const headers: Record<string, string> = { Accept: "application/json" };
  const token = deps.getToken();
  if (token !== null) headers.Authorization = `Bearer ${token}`;
  const init: RequestInit = { method: options.method ?? "GET", headers, credentials: "omit", cache: "no-store" };
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }

  let response: Response;
  try {
    response = await deps.fetch(`${deps.baseUrl()}${path}`, init);
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError("NETWORK");
  }

  if (!response.ok) {
    if (response.status === 401) {
      deps.clearSession();
      throw new ApiError("UNAUTHORIZED", 401);
    }
    if (response.status === 403) throw new ApiError("FORBIDDEN", 403);
    if (response.status === 404) throw new ApiError("NOT_FOUND", 404);
    if (response.status === 409) throw new ApiError("CONFLICT", 409);
    if (response.status === 400 || response.status === 422) throw new ApiError("BAD_REQUEST", response.status);
    throw new ApiError("SERVER_ERROR", response.status);
  }

  try {
    return parse(await response.json());
  } catch {
    throw new ApiError("INVALID_RESPONSE", response.status);
  }
}

// ------------------------------------------------------------------ the signed-in user

export interface CurrentUser {
  id: string;
  email: string;
  name: string | null;
  /** The linked GitHub account's login, if any. */
  githubLogin: string | null;
  /** True when the backend holds a GitHub token for this user. */
  githubConnected: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function text(value: unknown, max = 320): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= max;
}

export function parseCurrentUser(value: unknown): CurrentUser {
  if (!isRecord(value)) throw new ApiError("INVALID_RESPONSE");
  const { id, email, name, github_login: login, github_connected: connected } = value;
  if (!text(id, 64) || !text(email)) throw new ApiError("INVALID_RESPONSE");
  if (name != null && !text(name, 255)) throw new ApiError("INVALID_RESPONSE");
  if (login != null && !text(login, 255)) throw new ApiError("INVALID_RESPONSE");
  if (connected != null && typeof connected !== "boolean") throw new ApiError("INVALID_RESPONSE");
  return { id, email, name: name ?? null, githubLogin: login ?? null, githubConnected: connected === true };
}

export function getCurrentUser(deps?: ApiDeps): Promise<CurrentUser> {
  return apiRequest("/api/v1/auth/me", parseCurrentUser, {}, deps);
}
