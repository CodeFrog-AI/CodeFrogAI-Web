"use client";

import { useId, useState, type FormEvent } from "react";

import { PageContainer } from "@/components/layout/PageContainer";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

const FIELD = "mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm placeholder:text-muted";

/**
 * Settings placeholders. Nothing is saved: the API key lives only in this component's memory
 * (never in localStorage or anywhere else) and is discarded on save and on leaving the page.
 */
export function SettingsPage() {
  const keyId = useId();
  const modelId = useId();
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [notice, setNotice] = useState("");

  function save(event: FormEvent) {
    event.preventDefault();
    setApiKey("");
    setNotice("Saving settings is not available yet. Nothing was stored.");
  }

  return (
    <PageContainer title="Settings" description="Configure the AI provider and your GitHub connection.">
      <form onSubmit={save} className="space-y-6">
        <Card title="AI Provider" description="The model CodeFrog uses to understand and change your code.">
          <div className="space-y-4">
            <div>
              <label htmlFor={keyId} className="text-sm font-medium">
                API key
              </label>
              <input
                id={keyId}
                type="password"
                autoComplete="off"
                spellCheck={false}
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                placeholder="Paste your provider API key"
                className={FIELD}
              />
              <p className="mt-1 text-xs text-muted">Not saved or sent anywhere in this preview.</p>
            </div>
            <div>
              <label htmlFor={modelId} className="text-sm font-medium">
                Model
              </label>
              <input
                id={modelId}
                type="text"
                autoComplete="off"
                value={model}
                onChange={(event) => setModel(event.target.value)}
                placeholder="e.g. gpt-4o-mini"
                className={FIELD}
              />
            </div>
          </div>
        </Card>

        <Card title="GitHub connection" description="Connect your account so CodeFrog can work with pull requests.">
          <p className="text-sm text-muted">Not connected. GitHub sign-in is not available in the desktop app yet.</p>
        </Card>

        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary">
            Save settings
          </Button>
          <p role="status" className="text-sm text-muted">
            {notice}
          </p>
        </div>
      </form>
    </PageContainer>
  );
}
