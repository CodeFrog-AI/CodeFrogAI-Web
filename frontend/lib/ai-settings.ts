/**
 * The signed-in user's AI provider settings, through the CodeFrog backend:
 *
 *   GET    /api/v1/settings/ai               what is configured (never a key)
 *   PUT    /api/v1/settings/ai               a partial update (the only request that carries a key)
 *   DELETE /api/v1/settings/ai/llm-key       remove the LLM key
 *   DELETE /api/v1/settings/ai/embedding-key remove the embedding key
 *
 * The backend never returns a key: it reports only whether one is configured and its last four
 * characters. A typed key lives only in the form's memory, is sent once in the PUT body (never
 * in a URL), and is cleared afterwards. The LLM and embedding sides are independent.
 */

import { apiRequest, ApiError, type ApiDeps } from "@/lib/api";

export const AI_SETTINGS_PATH = "/api/v1/settings/ai";
export const LLM_KEY_PATH = `${AI_SETTINGS_PATH}/llm-key`;
export const EMBEDDING_KEY_PATH = `${AI_SETTINGS_PATH}/embedding-key`;
export const EMBEDDING_MODEL_CHANGED_NOTICE = "Embedding model changed. Re-scan your repositories to rebuild semantic embeddings.";

export type KeySource = "user" | "server" | "none";

export interface ProviderSettings {
  apiKeyConfigured: boolean;
  /** The last four characters of the saved key, for display only. */
  apiKeyHint: string | null;
  model: string;
  /** Whose key is used: the user's own, the server's, or none. */
  source: KeySource;
}

export interface AISettings {
  llm: ProviderSettings;
  embedding: ProviderSettings;
}

/** What the form holds. Keys are only ever what the user typed since the last load or save. */
export interface AISettingsForm {
  llmApiKey: string;
  llmModel: string;
  embeddingApiKey: string;
  embeddingModel: string;
}

/** The JSON body of a PUT: only the fields that changed. */
export interface AISettingsUpdate {
  llm_api_key?: string;
  llm_model?: string;
  embedding_api_key?: string;
  embedding_model?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseProvider(value: unknown): ProviderSettings {
  if (!isRecord(value)) throw new ApiError("INVALID_RESPONSE");
  const { api_key_configured: configured, api_key_hint: hint, model, source } = value;
  if (typeof configured !== "boolean") throw new ApiError("INVALID_RESPONSE");
  if (hint !== null && (typeof hint !== "string" || hint.length > 8)) throw new ApiError("INVALID_RESPONSE");
  if (typeof model !== "string" || model.length > 128) throw new ApiError("INVALID_RESPONSE");
  if (source !== "user" && source !== "server" && source !== "none") throw new ApiError("INVALID_RESPONSE");
  return { apiKeyConfigured: configured, apiKeyHint: hint, model, source };
}

export function parseAISettings(value: unknown): AISettings {
  if (!isRecord(value)) throw new ApiError("INVALID_RESPONSE");
  return { llm: parseProvider(value.llm), embedding: parseProvider(value.embedding) };
}

export function getAISettings(deps?: ApiDeps): Promise<AISettings> {
  return apiRequest(AI_SETTINGS_PATH, parseAISettings, {}, deps);
}

export function saveAISettings(update: AISettingsUpdate, deps?: ApiDeps): Promise<AISettings> {
  return apiRequest(AI_SETTINGS_PATH, parseAISettings, { method: "PUT", body: update }, deps);
}

export function removeAIKey(side: "llm" | "embedding", deps?: ApiDeps): Promise<AISettings> {
  return apiRequest(side === "llm" ? LLM_KEY_PATH : EMBEDDING_KEY_PATH, parseAISettings, { method: "DELETE" }, deps);
}

// ------------------------------------------------------------------ the form

/** The form for freshly loaded settings: keys are never pre-filled, models show what is in effect. */
export function formFromSettings(settings: AISettings): AISettingsForm {
  return { llmApiKey: "", llmModel: settings.llm.model, embeddingApiKey: "", embeddingModel: settings.embedding.model };
}

/** A key is sent only when something was typed; a model only when it differs from what is saved. */
export function buildUpdate(form: AISettingsForm, loaded: AISettings): AISettingsUpdate {
  const update: AISettingsUpdate = {};
  if (form.llmApiKey.trim() !== "") update.llm_api_key = form.llmApiKey.trim();
  if (form.embeddingApiKey.trim() !== "") update.embedding_api_key = form.embeddingApiKey.trim();
  if (form.llmModel.trim() !== loaded.llm.model) update.llm_model = form.llmModel.trim();
  if (form.embeddingModel.trim() !== loaded.embedding.model) update.embedding_model = form.embeddingModel.trim();
  return update;
}

export function hasChanges(update: AISettingsUpdate): boolean {
  return Object.keys(update).length > 0;
}

/** True when saving changed the embedding model in effect (existing embeddings no longer match it). */
export function embeddingModelChanged(before: AISettings, after: AISettings): boolean {
  return before.embedding.model !== after.embedding.model;
}
