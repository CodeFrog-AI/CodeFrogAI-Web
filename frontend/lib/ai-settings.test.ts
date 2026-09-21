import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it, vi } from "vitest";

import {
  AI_SETTINGS_PATH,
  buildUpdate,
  EMBEDDING_KEY_PATH,
  EMBEDDING_MODEL_CHANGED_NOTICE,
  embeddingModelChanged,
  formFromSettings,
  getAISettings,
  hasChanges,
  LLM_KEY_PATH,
  parseAISettings,
  removeAIKey,
  saveAISettings,
  type AISettings,
} from "@/lib/ai-settings";
import { ApiError, type ApiDeps } from "@/lib/api";

const JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl";
const KEY = "sk-typed-secret-key-1234";

const RAW = {
  llm: { api_key_configured: true, api_key_hint: "1234", model: "gpt-4o-mini", source: "user" },
  embedding: { api_key_configured: false, api_key_hint: null, model: "text-embedding-3-small", source: "server" },
};
const LOADED: AISettings = parseAISettings(RAW);

function deps(response: Response) {
  const fetchMock = vi.fn<ApiDeps["fetch"]>(async () => response);
  const clear = vi.fn();
  const value: ApiDeps = { fetch: fetchMock, baseUrl: () => "http://localhost:8000", getToken: () => JWT, clearSession: clear };
  return { value, fetchMock, clear };
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

async function failureOf(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    return error as ApiError;
  }
  throw new Error("expected a failure");
}

describe("parsing", () => {
  it("maps the backend response and never carries a key", () => {
    expect(LOADED.llm).toEqual({ apiKeyConfigured: true, apiKeyHint: "1234", model: "gpt-4o-mini", source: "user" });
    expect(LOADED.embedding.apiKeyConfigured).toBe(false);
    expect(JSON.stringify(LOADED)).not.toMatch(/sk-/);
  });

  it("rejects malformed responses", () => {
    expect(() => parseAISettings(null)).toThrow(ApiError);
    expect(() => parseAISettings({ llm: RAW.llm })).toThrow(ApiError);
    expect(() => parseAISettings({ ...RAW, llm: { ...RAW.llm, source: "other" } })).toThrow(ApiError);
    expect(() => parseAISettings({ ...RAW, llm: { ...RAW.llm, api_key_configured: "yes" } })).toThrow(ApiError);
  });
});

describe("requests", () => {
  it("loads the settings with a GET to the settings endpoint only", async () => {
    const d = deps(json(RAW));
    expect(await getAISettings(d.value)).toEqual(LOADED);
    const [url, init] = d.fetchMock.mock.calls[0];
    expect(url).toBe(`http://localhost:8000${AI_SETTINGS_PATH}`);
    expect(init.method).toBe("GET");
    expect(init.body).toBeUndefined();
  });

  it("sends a key only in the PUT body, never in the URL", async () => {
    const d = deps(json(RAW));
    await saveAISettings({ llm_api_key: KEY }, d.value);
    const [url, init] = d.fetchMock.mock.calls[0];
    expect(url).toBe(`http://localhost:8000${AI_SETTINGS_PATH}`);
    expect(url).not.toContain(KEY);
    expect(url).not.toContain("?");
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ llm_api_key: KEY });
    expect(init.credentials).toBe("omit");
  });

  it("removes each key with its own DELETE and no body", async () => {
    const llm = deps(json(RAW));
    await removeAIKey("llm", llm.value);
    expect(llm.fetchMock.mock.calls[0][0]).toBe(`http://localhost:8000${LLM_KEY_PATH}`);
    expect(llm.fetchMock.mock.calls[0][1].method).toBe("DELETE");
    expect(llm.fetchMock.mock.calls[0][1].body).toBeUndefined();

    const embedding = deps(json(RAW));
    await removeAIKey("embedding", embedding.value);
    expect(embedding.fetchMock.mock.calls[0][0]).toBe(`http://localhost:8000${EMBEDDING_KEY_PATH}`);
  });

  it("keeps the current 401 and 403 behavior", async () => {
    const unauthorized = deps(json({}, 401));
    expect((await failureOf(getAISettings(unauthorized.value))).code).toBe("UNAUTHORIZED");
    expect(unauthorized.clear).toHaveBeenCalledOnce();
    const forbidden = deps(json({}, 403));
    expect((await failureOf(saveAISettings({ llm_model: "m" }, forbidden.value))).code).toBe("FORBIDDEN");
    expect(forbidden.clear).not.toHaveBeenCalled();
  });

  it("does not put the key in error messages", async () => {
    const d = deps(new Response(`echo ${KEY}`, { status: 500 }));
    const error = await failureOf(saveAISettings({ llm_api_key: KEY }, d.value));
    expect(error.message).not.toContain(KEY);
  });
});

describe("the form", () => {
  it("has state for four fields, with keys never pre-filled", () => {
    expect(formFromSettings(LOADED)).toEqual({
      llmApiKey: "",
      llmModel: "gpt-4o-mini",
      embeddingApiKey: "",
      embeddingModel: "text-embedding-3-small",
    });
  });

  it("sends nothing when nothing changed", () => {
    expect(hasChanges(buildUpdate(formFromSettings(LOADED), LOADED))).toBe(false);
  });

  it("saves partially: only the LLM key", () => {
    const form = { ...formFromSettings(LOADED), llmApiKey: KEY };
    expect(buildUpdate(form, LOADED)).toEqual({ llm_api_key: KEY });
  });

  it("saves partially: only the embedding model", () => {
    const form = { ...formFromSettings(LOADED), embeddingModel: "text-embedding-3-large" };
    expect(buildUpdate(form, LOADED)).toEqual({ embedding_model: "text-embedding-3-large" });
  });

  it("keeps the two sides independent, including the same key on both", () => {
    const form = { ...formFromSettings(LOADED), llmApiKey: KEY, embeddingApiKey: KEY };
    expect(buildUpdate(form, LOADED)).toEqual({ llm_api_key: KEY, embedding_api_key: KEY });
    expect(buildUpdate({ ...formFromSettings(LOADED), embeddingApiKey: "other-key-5678" }, LOADED)).toEqual({ embedding_api_key: "other-key-5678" });
  });

  it("treats a blank key as unchanged and a cleared model as a reset", () => {
    const form = { ...formFromSettings(LOADED), llmApiKey: "   ", llmModel: "" };
    expect(buildUpdate(form, LOADED)).toEqual({ llm_model: "" });
  });

  it("clears the keys after a save or load", () => {
    const saved = parseAISettings({ ...RAW, embedding: { ...RAW.embedding, api_key_configured: true, api_key_hint: "5678", source: "user" } });
    expect(formFromSettings(saved).llmApiKey).toBe("");
    expect(formFromSettings(saved).embeddingApiKey).toBe("");
  });

  it("notices when the embedding model changed", () => {
    const after = parseAISettings({ ...RAW, embedding: { ...RAW.embedding, model: "text-embedding-3-large" } });
    expect(embeddingModelChanged(LOADED, after)).toBe(true);
    expect(embeddingModelChanged(LOADED, LOADED)).toBe(false);
    expect(EMBEDDING_MODEL_CHANGED_NOTICE).toBe("Embedding model changed. Re-scan your repositories to rebuild semantic embeddings.");
  });
});

describe("the settings page", () => {
  const page = readFileSync(fileURLToPath(new URL("../components/pages/SettingsPage.tsx", import.meta.url)), "utf8");

  it("has the four labelled fields, with password inputs for the keys and no autofill", () => {
    for (const label of ["LLM API Key", "LLM Model", "Embedding API Key", "Embedding Model"]) expect(page).toContain(label);
    expect(page.match(/type="password"/g)).toHaveLength(2);
    expect(page.match(/autoComplete="off"/g)?.length).toBeGreaterThanOrEqual(5);
    expect(page).toContain("Save Settings");
    expect(page).toContain("Remove key");
    expect(page).not.toMatch(/base.?url/i);
  });

  it("never pre-fills or stores a key, and uses no browser storage", () => {
    expect(page).not.toMatch(/localStorage|sessionStorage|indexedDB|document\.cookie|console\./);
    expect(page).not.toMatch(/llmApiKey:\s*settings|value=\{settings/);
  });

  it("makes no request of its own", () => {
    expect(page).not.toMatch(/\bfetch\(|apiRequest\(/);
  });
});
