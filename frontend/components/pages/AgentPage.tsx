"use client";

import { useId, useState, type FormEvent } from "react";

import { PageContainer } from "@/components/layout/PageContainer";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";

export function AgentPage() {
  const inputId = useId();
  const [draft, setDraft] = useState("");
  const [notice, setNotice] = useState("");

  function send(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim()) return;
    // The agent is not connected in this phase: nothing is sent anywhere.
    setNotice("The AI agent is not connected yet, so this message was not sent.");
  }

  return (
    <PageContainer title="AI Agent" description="The CodeFrog AI Agent workspace. Ask questions about your code and, later, plan and implement changes.">
      <Card title="Conversation">
        <div className="min-h-64" aria-live="polite">
          <EmptyState
            icon="agent"
            title="No messages yet"
            description="Agent messages will appear here. Try asking how a part of your repository works."
          />
        </div>
      </Card>

      <form onSubmit={send} className="space-y-3">
        <label htmlFor={inputId} className="block text-sm font-medium">
          Message to the agent
        </label>
        <textarea
          id={inputId}
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
            setNotice("");
          }}
          rows={3}
          placeholder="Ask about your repository…"
          className="w-full resize-y rounded-md border border-border bg-surface px-3 py-2 text-sm placeholder:text-muted"
        />
        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary" disabled={!draft.trim()}>
            Send
          </Button>
          <p role="status" className="text-sm text-muted">
            {notice}
          </p>
        </div>
      </form>
    </PageContainer>
  );
}
