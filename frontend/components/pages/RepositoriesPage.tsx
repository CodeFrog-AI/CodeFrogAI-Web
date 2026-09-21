"use client";

import { useState } from "react";

import { PageContainer } from "@/components/layout/PageContainer";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { useApp } from "@/lib/app-context";
import type { RepositoryInfo } from "@/lib/app-state";
import { selectLocalRepository, toRepositoryError, type LocalRepositoryError } from "@/lib/local-repository";
import { describeBranch, describeGitStatus, describeRemote } from "@/lib/repository";

type OpenState = { kind: "idle" } | { kind: "loading" } | { kind: "error"; error: LocalRepositoryError };

/** Runs the native "open repository" flow and tracks its loading and error state. */
function useOpenRepository() {
  const { dispatch } = useApp();
  const [state, setState] = useState<OpenState>({ kind: "idle" });

  async function open() {
    setState({ kind: "loading" });
    try {
      const result = await selectLocalRepository();
      if (result.status === "selected") {
        dispatch({ type: "select-repository", repository: result.repository });
      }
      setState({ kind: "idle" });
    } catch (error) {
      setState({ kind: "error", error: toRepositoryError(error) });
    }
  }

  return { state, open };
}

export function RepositoriesPage() {
  const { state } = useApp();
  return state.repository ? <RepositoryDashboard repository={state.repository} /> : <Welcome />;
}

function OpenRepositoryError({ error, onRetry }: { error: LocalRepositoryError; onRetry: () => void }) {
  return (
    <div role="alert" className="max-w-md rounded-md border border-danger/60 bg-surface px-4 py-3 text-left text-sm">
      <p className="font-medium text-danger">Could not open the repository</p>
      <p className="mt-1 text-muted">{error.message}</p>
      {error.code !== "DESKTOP_REQUIRED" && (
        <Button className="mt-3" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

function Welcome() {
  const { dispatch } = useApp();
  const { state, open } = useOpenRepository();
  const loading = state.kind === "loading";

  return (
    <PageContainer
      title="Welcome to CodeFrog"
      description="A local-first AI software engineer. Understand a codebase, plan changes, review diffs, and ship pull requests, with you approving every step."
    >
      <EmptyState
        title="No repository selected"
        description="Open a local Git repository to get started. CodeFrog only reads its name, branch, and status for now."
      >
        <Button variant="primary" onClick={open} disabled={loading} aria-busy={loading}>
          {loading ? "Opening…" : "Open Repository"}
        </Button>
        <Button onClick={() => dispatch({ type: "navigate", page: "settings" })}>Connect GitHub</Button>
      </EmptyState>
      <p role="status" className="sr-only">
        {loading ? "Opening repository" : ""}
      </p>
      {state.kind === "error" && <OpenRepositoryError error={state.error} onRetry={open} />}
    </PageContainer>
  );
}

function RepositoryDashboard({ repository }: { repository: RepositoryInfo }) {
  const { dispatch } = useApp();
  const { state, open } = useOpenRepository();
  const [notice, setNotice] = useState("");
  const loading = state.kind === "loading";

  return (
    <PageContainer title={repository.name} description="Local repository">
      <Card title="Repository">
        <dl className="grid gap-4 text-sm sm:grid-cols-2">
          <Detail label="Name" value={repository.name} />
          <Detail label="Local path" value={repository.path} mono />
          <Detail label="Current branch" value={describeBranch(repository.branch)} mono />
          <Detail label="Git status" value={describeGitStatus(repository.isDirty)} />
          <Detail label="Remote URL" value={describeRemote(repository.remoteUrl)} mono={repository.remoteUrl !== null} />
        </dl>
      </Card>

      <Card title="Quick actions">
        <div className="flex flex-wrap gap-3">
          <Button variant="primary" onClick={() => dispatch({ type: "navigate", page: "agent" })}>
            Ask Agent
          </Button>
          <Button onClick={() => setNotice("Repository analysis is not available yet.")}>Analyze Repository</Button>
          <Button onClick={() => dispatch({ type: "navigate", page: "pull-requests" })}>View Pull Requests</Button>
          <Button onClick={open} disabled={loading} aria-busy={loading}>
            {loading ? "Opening…" : "Open another repository"}
          </Button>
          <Button variant="ghost" onClick={() => dispatch({ type: "clear-repository" })}>
            Close repository
          </Button>
        </div>
        <p role="status" className="mt-3 min-h-5 text-sm text-muted">
          {notice}
        </p>
        {state.kind === "error" && <OpenRepositoryError error={state.error} onRetry={open} />}
      </Card>
    </PageContainer>
  );
}

function Detail({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="text-muted">{label}</dt>
      <dd className={`mt-0.5 break-all ${mono ? "font-mono" : ""}`}>{value}</dd>
    </div>
  );
}
