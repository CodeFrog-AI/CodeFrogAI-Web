import { describe, expect, it, vi } from "vitest";

import {
  API_ERROR_MESSAGES,
  apiRequest,
  ApiError,
  DEFAULT_API_URL,
  getCurrentUser,
  githubLoginUrl,
  normalizeBaseUrl,
  parseCurrentUser,
  type ApiDeps,
  type ApiErrorCode,
} from "@/lib/api";

const BASE = "http://localhost:8000";
const JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl";
const LEAK = "Traceback (most recent call last): sqlalchemy.exc.OperationalError gho_secrettoken";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function deps(response: Response | (() => Promise<Response>), overrides: Partial<ApiDeps> = {}) {
  const fetchMock = vi.fn<ApiDeps["fetch"]>(async () => (typeof response === "function" ? response() : response));
  const clear = vi.fn();
  const value: ApiDeps = { fetch: fetchMock, baseUrl: () => BASE, getToken: () => JWT, clearSession: clear, ...overrides };
  return { value, fetchMock, clear };
}

async function failureOf(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    return error as ApiError;
  }
  throw new Error("expected the request to fail");
}

const identity = (value: unknown) => value;

describe("normalizeBaseUrl", () => {
  it("defaults to the local development backend", () => {
    expect(normalizeBaseUrl(undefined)).toBe(DEFAULT_API_URL);
    expect(normalizeBaseUrl("  ")).toBe(DEFAULT_API_URL);
  });

  it.each([
    ["http://localhost:8000", "http://localhost:8000"],
    ["http://localhost:8000/", "http://localhost:8000"],
    ["https://api.example.com", "https://api.example.com"],
  ])("accepts %s", (input, expected) => {
    expect(normalizeBaseUrl(input)).toBe(expected);
  });

  it.each([
    "not a url",
    "ftp://example.com",
    "javascript:alert(1)",
    "http://user:pass@localhost:8000",
    "http://localhost:8000/api",
    "http://localhost:8000?x=1",
    "http://localhost:8000#frag",
  ])("rejects %s", (input) => {
    expect(() => normalizeBaseUrl(input)).toThrowError(expect.objectContaining({ code: "CONFIG" }));
  });
});

describe("githubLoginUrl", () => {
  it("is the backend's GitHub login endpoint", () => {
    expect(githubLoginUrl(BASE)).toBe("http://localhost:8000/api/v1/auth/github/login");
    expect(githubLoginUrl("https://api.example.com")).toBe("https://api.example.com/api/v1/auth/github/login");
  });
});

describe("apiRequest", () => {
  it("sends the session JWT as a Bearer header and nothing else as credentials", async () => {
    const { value, fetchMock } = deps(json({ ok: true }));
    await apiRequest("/api/v1/auth/me", identity, {}, value);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/v1/auth/me");
    expect((init.headers as Record<string, string>).Authorization).toBe(`Bearer ${JWT}`);
    expect(init.credentials).toBe("omit");
    expect(init.method).toBe("GET");
  });

  it("sends no Authorization header without a session", async () => {
    const { value, fetchMock } = deps(json({}), { getToken: () => null });
    await apiRequest("/api/v1/auth/me", identity, {}, value);
    expect((fetchMock.mock.calls[0][1].headers as Record<string, string>).Authorization).toBeUndefined();
  });

  it("sends a JSON body for POST requests", async () => {
    const { value, fetchMock } = deps(json({}));
    await apiRequest("/api/v1/repositories/connect", identity, { method: "POST", body: { github_repository_id: 5 } }, value);
    const init = fetchMock.mock.calls[0][1];
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify({ github_repository_id: 5 }));
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
  });

  it("returns the parsed response", async () => {
    const { value } = deps(json({ answer: 42 }));
    expect(await apiRequest("/api/v1/x", (v) => (v as { answer: number }).answer, {}, value)).toBe(42);
  });

  it.each(["https://evil.example/api/v1/x", "//evil.example/api/v1/x", "/other/path", "/api/v1/../admin", "/api/v1//x", "api/v1/x", "/api/v1/x?y=1", "/api/v1/x#y", ""])(
    "refuses the path %j without calling fetch",
    async (path) => {
      const { value, fetchMock } = deps(json({}));
      const error = await failureOf(apiRequest(path, identity, {}, value));
      expect(error.code).toBe("CONFIG");
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("clears the session on 401 and reports UNAUTHORIZED", async () => {
    const { value, clear } = deps(json({ detail: LEAK }, 401));
    const error = await failureOf(apiRequest("/api/v1/auth/me", identity, {}, value));
    expect(error.code).toBe("UNAUTHORIZED");
    expect(error.status).toBe(401);
    expect(clear).toHaveBeenCalledOnce();
  });

  it("reports 403 as FORBIDDEN without clearing the session", async () => {
    const { value, clear } = deps(json({ detail: LEAK }, 403));
    const error = await failureOf(apiRequest("/api/v1/repositories/github", identity, {}, value));
    expect(error.code).toBe("FORBIDDEN");
    expect(error.message).toContain("Reconnect GitHub");
    expect(clear).not.toHaveBeenCalled();
  });

  it.each([
    [404, "NOT_FOUND"],
    [409, "CONFLICT"],
    [400, "BAD_REQUEST"],
    [422, "BAD_REQUEST"],
    [429, "SERVER_ERROR"],
    [500, "SERVER_ERROR"],
    [502, "SERVER_ERROR"],
    [503, "SERVER_ERROR"],
  ] satisfies [number, ApiErrorCode][])("maps HTTP %d to %s", async (status, code) => {
    const { value } = deps(json({ detail: LEAK }, status));
    expect((await failureOf(apiRequest("/api/v1/x", identity, {}, value))).code).toBe(code);
  });

  it("never exposes the backend's error body", async () => {
    for (const status of [400, 401, 403, 404, 409, 422, 500, 502]) {
      const { value } = deps(json({ detail: LEAK, error: { message: LEAK } }, status));
      const error = await failureOf(apiRequest("/api/v1/x", identity, {}, value));
      expect(error.message).toBe(API_ERROR_MESSAGES[error.code]);
      expect(error.message).not.toMatch(/Traceback|sqlalchemy|gho_/);
    }
  });

  it("reports an unreachable backend as NETWORK", async () => {
    const { value } = deps(async () => {
      throw new TypeError("Failed to fetch http://localhost:8000 with token " + JWT);
    });
    const error = await failureOf(apiRequest("/api/v1/x", identity, {}, value));
    expect(error.code).toBe("NETWORK");
    expect(error.message).not.toContain(JWT);
  });

  it("reports a non-JSON success response as INVALID_RESPONSE", async () => {
    const { value } = deps(new Response("<html>oops</html>", { status: 200 }));
    expect((await failureOf(apiRequest("/api/v1/x", identity, {}, value))).code).toBe("INVALID_RESPONSE");
  });

  it("reports a response that fails validation as INVALID_RESPONSE", async () => {
    const { value } = deps(json({ nope: true }));
    const error = await failureOf(
      apiRequest(
        "/api/v1/x",
        () => {
          throw new Error("bad shape");
        },
        {},
        value,
      ),
    );
    expect(error.code).toBe("INVALID_RESPONSE");
  });
});

describe("getCurrentUser", () => {
  const ME = { id: "u1", email: "a@example.com", name: "Ada", status: "active", created_at: "2026-01-01T00:00:00Z", github_login: "octocat", github_connected: true };

  it("calls GET /api/v1/auth/me and returns the user with the GitHub connection", async () => {
    const { value, fetchMock } = deps(json(ME));
    expect(await getCurrentUser(value)).toEqual({ id: "u1", email: "a@example.com", name: "Ada", githubLogin: "octocat", githubConnected: true });
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/api/v1/auth/me");
  });

  it("treats a missing GitHub link as not connected", () => {
    expect(parseCurrentUser({ id: "u1", email: "a@example.com", name: null })).toEqual({
      id: "u1",
      email: "a@example.com",
      name: null,
      githubLogin: null,
      githubConnected: false,
    });
  });

  it.each([
    ["null", null],
    ["an array", []],
    ["a missing id", { email: "a@example.com" }],
    ["a missing email", { id: "u1" }],
    ["a numeric login", { id: "u1", email: "a@example.com", github_login: 5 }],
    ["a string connected flag", { id: "u1", email: "a@example.com", github_connected: "yes" }],
  ])("rejects %s", (_label, value) => {
    expect(() => parseCurrentUser(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });

  it("never exposes a token from the response", () => {
    const user = parseCurrentUser({ ...ME, access_token: "gho_secret", github_access_token: "gho_secret" });
    expect(JSON.stringify(user)).not.toContain("gho_");
  });
});
