import { describe, expect, it } from "vitest";

import { clearSession, getSessionToken, hasSession, isValidToken, SESSION_KEY, setSessionToken, type TokenStorage } from "@/lib/session";

const JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl";

function fakeStorage(initial: Record<string, string> = {}): TokenStorage & { data: Record<string, string> } {
  const data = { ...initial };
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

const throwingStorage: TokenStorage = {
  getItem: () => {
    throw new Error("blocked");
  },
  setItem: () => {
    throw new Error("blocked");
  },
  removeItem: () => {
    throw new Error("blocked");
  },
};

describe("isValidToken", () => {
  it("accepts a JWT-shaped token", () => {
    expect(isValidToken(JWT)).toBe(true);
  });

  it.each([
    ["an empty string", ""],
    ["whitespace", "   "],
    ["a token with a space", "abc def"],
    ["a token with a newline", "abc" + String.fromCharCode(10) + "def"],
    ["a token with a NUL", "abc" + String.fromCharCode(0)],
    ["markup", "<script>alert(1)</script>"],
    ["a quote", 'abc"def'],
    ["a non-string", 42],
    ["null", null],
    ["an over-long value", "a".repeat(5000)],
  ])("rejects %s", (_label, value) => {
    expect(isValidToken(value)).toBe(false);
  });
});

describe("session storage", () => {
  it("stores, reads, and clears the token", () => {
    const storage = fakeStorage();
    expect(hasSession(storage)).toBe(false);
    expect(setSessionToken(JWT, storage)).toBe(true);
    expect(getSessionToken(storage)).toBe(JWT);
    expect(hasSession(storage)).toBe(true);
    clearSession(storage);
    expect(getSessionToken(storage)).toBeNull();
    expect(hasSession(storage)).toBe(false);
  });

  it("keeps the token under one key and stores nothing else", () => {
    const storage = fakeStorage();
    setSessionToken(JWT, storage);
    expect(Object.keys(storage.data)).toEqual([SESSION_KEY]);
    expect(storage.data[SESSION_KEY]).toBe(JWT);
  });

  it("refuses to store an invalid token", () => {
    const storage = fakeStorage();
    expect(setSessionToken("not a token!", storage)).toBe(false);
    expect(setSessionToken("", storage)).toBe(false);
    expect(storage.data).toEqual({});
  });

  it("does not trust a malformed value already in storage", () => {
    const storage = fakeStorage({ [SESSION_KEY]: "<b>evil</b>" });
    expect(getSessionToken(storage)).toBeNull();
    expect(hasSession(storage)).toBe(false);
  });

  it("does not throw when storage is blocked or missing", () => {
    expect(getSessionToken(throwingStorage)).toBeNull();
    expect(setSessionToken(JWT, throwingStorage)).toBe(false);
    expect(() => clearSession(throwingStorage)).not.toThrow();
    expect(getSessionToken(null)).toBeNull();
    expect(setSessionToken(JWT, null)).toBe(false);
    expect(() => clearSession(null)).not.toThrow();
  });
});
