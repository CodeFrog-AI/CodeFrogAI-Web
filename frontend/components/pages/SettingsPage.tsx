"use client";

import { useCallback, useEffect, useId, useState, type FormEvent } from "react";

import { GitHubConnectionCard } from "@/components/github/GitHubConnectionCard";
import { PageContainer } from "@/components/layout/PageContainer";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ApiError } from "@/lib/api";
import {
  buildUpdate,
  EMBEDDING_MODEL_CHANGED_NOTICE,
  embeddingModelChanged,
  formFromSettings,
  getAISettings,
  hasChanges,
  removeAIKey,
  saveAISettings,
  type AISettings,
  type AISettingsForm,
  type ProviderSettings,
} from "@/lib/ai-settings";
import { useAuth } from "@/lib/auth-context";
import { connectGitHub } from "@/lib/github-auth";

const FIELD = "mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm placeholder:text-muted";

type Load = { status: "loading" } | { status: "ready"; settings: AISettings } | { status: "error"; error: ApiError };

function asApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError("SERVER_ERROR");
}

function describeKey(provider: ProviderSettings): string {
  if (provider.source === "user") return `Saved key ending in ${provider.apiKeyHint ?? "····"}.`;
  if (provider.source === "server") return "No key of your own is saved: the server default key is used.";
  return "No key is configured.";
}

/**
 * The user's own AI provider settings, kept as two independent sections. Typed keys live only
 * in this component's memory: they are never pre-filled, never written to browser storage, sent
 * only to the settings endpoint, and cleared as soon as they are saved.
 */
function AISettingsPanel() {
  const { expire } = useAuth();
  const ids = { llmKey: useId(), llmModel: useId(), embeddingKey: useId(), embeddingModel: useId() };
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [form, setForm] = useState<AISettingsForm>({ llmApiKey: "", llmModel: "", embeddingApiKey: "", embeddingModel: "" });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string } | null>(null);
  const [reload, setReload] = useState(0);

  const fail = useCallback(
    (error: unknown) => {
      const apiError = asApiError(error);
      if (apiError.code === "UNAUTHORIZED") expire();
      return apiError;
    },
    [expire],
  );

  useEffect(() => {
    let cancelled = false;
    getAISettings()
      .then((settings) => {
        if (cancelled) return;
        setForm(formFromSettings(settings));
        setLoad({ status: "ready", settings });
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoad({ status: "error", error: fail(error) });
      });
    return () => {
      cancelled = true;
    };
  }, [reload, fail]);

  function retry() {
    setLoad({ status: "loading" });
    setReload((value) => value + 1);
  }

  function applied(before: AISettings, after: AISettings, text: string) {
    setForm(formFromSettings(after)); // keys are always cleared
    setLoad({ status: "ready", settings: after });
    setMessage({ tone: "ok", text: embeddingModelChanged(before, after) ? `${text} ${EMBEDDING_MODEL_CHANGED_NOTICE}` : text });
  }

  function save(event: FormEvent) {
    event.preventDefault();
    if (load.status !== "ready" || busy) return;
    const before = load.settings;
    const update = buildUpdate(form, before);
    if (!hasChanges(update)) {
      setMessage({ tone: "ok", text: "Nothing to save." });
      return;
    }
    setBusy(true);
    setMessage(null);
    saveAISettings(update)
      .then((after) => applied(before, after, "Settings saved."))
      .catch((error: unknown) => setMessage({ tone: "error", text: fail(error).message }))
      .finally(() => setBusy(false));
  }

  function remove(side: "llm" | "embedding") {
    if (load.status !== "ready" || busy) return;
    const before = load.settings;
    setBusy(true);
    setMessage(null);
    removeAIKey(side)
      .then((after) => applied(before, after, "Key removed."))
      .catch((error: unknown) => setMessage({ tone: "error", text: fail(error).message }))
      .finally(() => setBusy(false));
  }

  if (load.status === "loading") {
    return (
      <p role="status" className="text-sm text-muted">
        Loading your AI settings…
      </p>
    );
  }

  if (load.status === "error") {
    return (
      <div role="alert" className="space-y-3 text-sm">
        <p className="text-muted">{load.error.message}</p>
        <Button onClick={retry}>Try again</Button>
      </div>
    );
  }

  const { settings } = load;
  return (
    <form onSubmit={save} className="space-y-6" autoComplete="off">
      <Card title="AI Provider" description="The model CodeFrog uses to understand and change your code.">
        <div className="space-y-4">
          <div>
            <label htmlFor={ids.llmKey} className="text-sm font-medium">
              LLM API Key
            </label>
            <input
              id={ids.llmKey}
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={form.llmApiKey}
              onChange={(event) => setForm({ ...form, llmApiKey: event.target.value })}
              placeholder={settings.llm.source === "user" ? "Enter a new key to replace it" : "Paste your provider API key"}
              className={FIELD}
            />
            <p className="mt-1 text-xs text-muted">{describeKey(settings.llm)}</p>
            {settings.llm.source === "user" && (
              <Button type="button" onClick={() => remove("llm")} disabled={busy}>
                Remove key
              </Button>
            )}
          </div>
          <div>
            <label htmlFor={ids.llmModel} className="text-sm font-medium">
              LLM Model
            </label>
            <input
              id={ids.llmModel}
              type="text"
              autoComplete="off"
              value={form.llmModel}
              onChange={(event) => setForm({ ...form, llmModel: event.target.value })}
              placeholder="e.g. gpt-4o-mini"
              className={FIELD}
            />
          </div>
        </div>
      </Card>

      <Card title="Embedding Provider" description="The model CodeFrog uses to index your code for semantic search. It has its own key.">
        <div className="space-y-4">
          <div>
            <label htmlFor={ids.embeddingKey} className="text-sm font-medium">
              Embedding API Key
            </label>
            <input
              id={ids.embeddingKey}
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={form.embeddingApiKey}
              onChange={(event) => setForm({ ...form, embeddingApiKey: event.target.value })}
              placeholder={settings.embedding.source === "user" ? "Enter a new key to replace it" : "Paste your embedding API key"}
              className={FIELD}
            />
            <p className="mt-1 text-xs text-muted">{describeKey(settings.embedding)}</p>
            {settings.embedding.source === "user" && (
              <Button type="button" onClick={() => remove("embedding")} disabled={busy}>
                Remove key
              </Button>
            )}
          </div>
          <div>
            <label htmlFor={ids.embeddingModel} className="text-sm font-medium">
              Embedding Model
            </label>
            <input
              id={ids.embeddingModel}
              type="text"
              autoComplete="off"
              value={form.embeddingModel}
              onChange={(event) => setForm({ ...form, embeddingModel: event.target.value })}
              placeholder="e.g. text-embedding-3-small"
              className={FIELD}
            />
          </div>
        </div>
      </Card>

      <div className="flex items-center gap-3">
        <Button type="submit" variant="primary" disabled={busy}>
          Save Settings
        </Button>
        {message !== null && (
          <p role={message.tone === "error" ? "alert" : "status"} className="text-sm text-muted">
            {message.text}
          </p>
        )}
      </div>
    </form>
  );
}

export function SettingsPage() {
  const { state } = useAuth();

  return (
    <PageContainer title="Settings" description="Configure the AI provider and your GitHub connection.">
      <div className="space-y-6">
        {state.status === "signed-in" && <AISettingsPanel />}
        {state.status === "signed-out" && (
          <Card title="AI Provider" description="Your own AI keys are saved to your CodeFrog account.">
            <div className="space-y-3 text-sm">
              <p className="text-muted">Connect GitHub to sign in and manage your AI provider settings.</p>
              <Button variant="primary" onClick={() => connectGitHub()}>
                Connect GitHub
              </Button>
            </div>
          </Card>
        )}
        <GitHubConnectionCard />
      </div>
    </PageContainer>
  );
}
