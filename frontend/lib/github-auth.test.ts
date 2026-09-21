import { describe, expect, it, vi } from "vitest";

import { ApiError, type CurrentUser } from "@/lib/api";
import { AUTH_ERROR_MESSAGES, completeSignIn, connectGitHub, parseCallbackHash, type AuthErrorCode } from "@/lib/github-auth";
import { SESSION_KEY, type TokenStorage } from "@/lib/session";

const JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl";
const USER: CurrentUser = { id: "u1", email: "octocat@example.com", name: "Octocat", githubLogin: "octocat", githubConnected: true };

function fakeStorage(): TokenStorage & { data: Record<string, string> } {
  const data: Record<string, string> = {};
  return {
    data,
    getItem: (key) => (key in data ? data[key] : null),
    setItem: (key, value) => {
      data[key] = value;
    },
    removeItem: (key) => {
      delete data[key];
    },
  };
}

describe("connectGitHub", () => {
  it("navigates the whole page to the backend's GitHub login endpoint", () => {
    const navigate = vi.fn();
    connectGitHub(navigate);
    expect(navigate).toHaveBeenCalledExactlyOnceWith("http://localhost:8000/api/v1/auth/github/login");
  });

  it("does not use fetch, popups, or iframes to start OAuth", () => {
    const originalFetch = globalThis.fetch;
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    try {
      connectGitHub(() => undefined);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe("parseCallbackHash", () => {
  it("extracts the access token", () => {
    expect(parseCallbackHash(`#access_token=${JWT}`)).toEqual({ kind: "token", token: JWT });
    expect(parseCallbackHash(`access_token=${JWT}`)).toEqual({ kind: "token", token: JWT });
  });

  it.each(["invalid_state", "access_denied", "authorization_failed", "account_unavailable", "server_error"] satisfies AuthErrorCode[])(
    "recognizes the error code %s",
    (code) => {
      expect(parseCallbackHash(`#error=${code}`)).toEqual({ kind: "error", code });
    },
  );

  it("turns an unknown or hostile error value into a generic failure", () => {
    expect(parseCallbackHash("#error=%3Cscript%3Ealert(1)%3C%2Fscript%3E")).toEqual({ kind: "error", code: "authorization_failed" });
    expect(parseCallbackHash("#error=Traceback")).toEqual({ kind: "error", code: "authorization_failed" });
    expect(parseCallbackHash("#error=storage_unavailable")).toEqual({ kind: "error", code: "authorization_failed" });
  });

  it("rejects a malformed token", () => {
    expect(parseCallbackHash("#access_token=")).toEqual({ kind: "error", code: "authorization_failed" });
    expect(parseCallbackHash("#access_token=has%20a%20space")).toEqual({ kind: "error", code: "authorization_failed" });
    expect(parseCallbackHash("#access_token=%3Cb%3E")).toEqual({ kind: "error", code: "authorization_failed" });
  });

  it("prefers a token over an error, and reports nothing for an empty hash", () => {
    expect(parseCallbackHash(`#access_token=${JWT}&error=x`).kind).toBe("token");
    expect(parseCallbackHash("")).toEqual({ kind: "empty" });
    expect(parseCallbackHash("#")).toEqual({ kind: "empty" });
    expect(parseCallbackHash("#unrelated=1")).toEqual({ kind: "empty" });
  });
});

describe("completeSignIn", () => {
  it("stores the token in session storage, cleans the URL, confirms with /auth/me, and signs in", async () => {
    const storage = fakeStorage();
    const events: string[] = [];
    const replaceUrl = vi.fn(() => events.push("replaceUrl"));
    const getCurrentUser = vi.fn(async () => {
      events.push("getCurrentUser");
      return USER;
    });

    const result = await completeSignIn(`#access_token=${JWT}`, { storage, replaceUrl, getCurrentUser });

    expect(result).toEqual({ status: "signed-in", user: USER });
    expect(storage.data).toEqual({ [SESSION_KEY]: JWT });
    expect(replaceUrl).toHaveBeenCalledOnce();
    expect(getCurrentUser).toHaveBeenCalledOnce();
    expect(events).toEqual(["replaceUrl", "getCurrentUser"]); // the URL is cleaned before anything slow happens
  });

  it("cleans the URL even when there is an error or nothing to read", async () => {
    for (const hash of ["#error=access_denied", "", "#access_token=bad token"]) {
      const replaceUrl = vi.fn();
      await completeSignIn(hash, { storage: fakeStorage(), replaceUrl, getCurrentUser: async () => USER });
      expect(replaceUrl, hash).toHaveBeenCalledOnce();
    }
  });

  it("reports an error from the backend without touching storage or calling /auth/me", async () => {
    const storage = fakeStorage();
    const getCurrentUser = vi.fn(async () => USER);

    const result = await completeSignIn("#error=access_denied", { storage, replaceUrl: () => undefined, getCurrentUser });

    expect(result).toEqual({ status: "failed", code: "access_denied" });
    expect(storage.data).toEqual({});
    expect(getCurrentUser).not.toHaveBeenCalled();
  });

  it("reports a missing fragment", async () => {
    const result = await completeSignIn("", { storage: fakeStorage(), replaceUrl: () => undefined, getCurrentUser: async () => USER });
    expect(result).toEqual({ status: "failed", code: "no_response" });
  });

  it("fails clearly when the browser blocks session storage", async () => {
    const blocked: TokenStorage = {
      getItem: () => null,
      setItem: () => {
        throw new Error("blocked");
      },
      removeItem: () => undefined,
    };
    const getCurrentUser = vi.fn(async () => USER);

    const result = await completeSignIn(`#access_token=${JWT}`, { storage: blocked, replaceUrl: () => undefined, getCurrentUser });

    expect(result).toEqual({ status: "failed", code: "storage_unavailable" });
    expect(getCurrentUser).not.toHaveBeenCalled();
  });

  it.each([
    ["NETWORK", "backend_unavailable"],
    ["UNAUTHORIZED", "session_rejected"],
    ["FORBIDDEN", "session_rejected"],
    ["SERVER_ERROR", "authorization_failed"],
    ["INVALID_RESPONSE", "authorization_failed"],
  ] as const)("maps an /auth/me failure (%s) to %s", async (apiCode, expected) => {
    const result = await completeSignIn(`#access_token=${JWT}`, {
      storage: fakeStorage(),
      replaceUrl: () => undefined,
      getCurrentUser: async () => {
        throw new ApiError(apiCode);
      },
    });
    expect(result).toEqual({ status: "failed", code: expected });
  });

  it("never puts the token in a result, message, or log", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    try {
      const failed = await completeSignIn(`#access_token=${JWT}`, {
        storage: fakeStorage(),
        replaceUrl: () => undefined,
        getCurrentUser: async () => {
          throw new ApiError("NETWORK");
        },
      });
      const ok = await completeSignIn(`#access_token=${JWT}`, { storage: fakeStorage(), replaceUrl: () => undefined, getCurrentUser: async () => USER });

      expect(JSON.stringify([failed, ok])).not.toContain(JWT);
      for (const spy of [log, warn, error]) expect(spy).not.toHaveBeenCalled();
      expect(Object.values(AUTH_ERROR_MESSAGES).join(" ")).not.toContain(JWT);
    } finally {
      log.mockRestore();
      warn.mockRestore();
      error.mockRestore();
    }
  });
});

describe("AUTH_ERROR_MESSAGES", () => {
  it("has a fixed friendly message for every code, with no raw text", () => {
    for (const [code, message] of Object.entries(AUTH_ERROR_MESSAGES)) {
      expect(message.length, code).toBeGreaterThan(10);
      expect(message).not.toMatch(/Traceback|exception|sql|token=|secret/i);
    }
  });
});
