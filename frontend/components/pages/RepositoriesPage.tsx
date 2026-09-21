"use client";

import { useState } from "react";

import { PageContainer } from "@/components/layout/PageContainer";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { GitHubRepositoryPicker } from "@/components/github/GitHubRepositoryPicker";
import { RepositoryExplorer } from "@/components/repository/RepositoryExplorer";
import { ApiError } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import type { GitHubRepository, LocalRepository } from "@/lib/app-state";
import { useAuth } from "@/lib/auth-context";
import { connectGitHub } from "@/lib/github-auth";
import { scanRepository, type ScanSummary } from "@/lib/github-repositories";
import { selectLocalRepository, toRepositoryError, type LocalRepositoryError } from "@/lib/local-repository";
import { describeBranch, describeGitStatus, describeRemote } from "@/lib/repository";
import { clearRepositorySelection } from "@/lib/repository-files";

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
  const repository = state.repository;
  if (repository?.source === "local") return <RepositoryDashboard repository={repository} />;
  if (repository?.source === "github") return <GitHubRepositoryDashboard repository={repository} />;
  return <Welcome />;
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
  const { state: auth } = useAuth();
  const { state, open } = useOpenRepository();
  const loading = state.kind === "loading";
  const githubReady = auth.status === "signed-in" && auth.user.githubConnected;

  return (
    <PageContainer
      title="Welcome to CodeFrog"
      description="Your AI developer partner for GitHub repositories. Understand a codebase, plan changes, review diffs, and ship pull requests, with you approving every step."
    >
      {githubReady ? (
        <GitHubRepositoryPicker />
      ) : (
        <EmptyState
          title="No repository selected"
          description="Connect GitHub to choose one of your repositories, or open a local Git repository in the CodeFrog desktop app."
        >
          <Button variant="primary" onClick={() => connectGitHub()} disabled={auth.status === "loading"}>
            Connect GitHub
          </Button>
          <Button onClick={open} disabled={loading} aria-busy={loading}>
            {loading ? "Opening…" : "Open Repository"}
          </Button>
        </EmptyState>
      )}

      {auth.status === "signed-out" && auth.expired && (
        <p role="status" className="text-sm text-muted">
          Your session has expired. Connect GitHub again to continue.
        </p>
      )}
      {auth.status === "error" && (
        <p role="alert" className="text-sm text-danger">
          {(auth.error instanceof ApiError ? auth.error : new ApiError("SERVER_ERROR")).message}
        </p>
      )}

      {githubReady && (
        <Card title="Local repository" description="Prefer a repository on this computer? Open it with the CodeFrog desktop app.">
          <Button onClick={open} disabled={loading} aria-busy={loading}>
            {loading ? "Opening…" : "Open Repository"}
          </Button>
        </Card>
      )}
      <p role="status" className="sr-only">
        {loading ? "Opening repository" : ""}
      </p>
      {state.kind === "error" && <OpenRepositoryError error={state.error} onRetry={open} />}
    </PageContainer>
  );
}

function RepositoryDashboard({ repository }: { repository: LocalRepository }) {
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
          <Button
            variant="ghost"
            onClick={() => {
              void clearRepositorySelection();
              dispatch({ type: "clear-repository" });
            }}
          >
            Close repository
          </Button>
        </div>
        <p role="status" className="mt-3 min-h-5 text-sm text-muted">
          {notice}
        </p>
        {state.kind === "error" && <OpenRepositoryError error={state.error} onRetry={open} />}
      </Card>

      {/* Keyed by path: opening another repository starts a fresh explorer. */}
      <RepositoryExplorer key={repository.path} />
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

function GitHubRepositoryDashboard({ repository }: { repository: GitHubRepository }) {
  const { dispatch } = useApp();
  const auth = useAuth();
  const [scan, setScan] = useState<{ kind: "idle" } | { kind: "running" } | { kind: "done"; summary: ScanSummary } | { kind: "error"; error: ApiError }>({
    kind: "idle",
  });

  async function runScan() {
    setScan({ kind: "running" });
    try {
      setScan({ kind: "done", summary: await scanRepository(repository.repositoryId) });
    } catch (error) {
      const apiError = error instanceof ApiError ? error : new ApiError("SERVER_ERROR");
      if (apiError.code === "UNAUTHORIZED") auth.expire();
      else setScan({ kind: "error", error: apiError });
    }
  }

  return (
    <PageContainer title={`${repository.owner}/${repository.name}`} description="Connected GitHub repository">
      <Card title="Repository">
        <dl className="grid gap-4 text-sm sm:grid-cols-2">
          <Detail label="Repository" value={`${repository.owner}/${repository.name}`} mono />
          <Detail label="Branch" value={repository.defaultBranch} mono />
          <Detail label="Source" value="GitHub" />
          <Detail label="Visibility" value={repository.private ? "Private" : "Public"} />
        </dl>
      </Card>

      <Card title="Quick actions">
        <div className="flex flex-wrap gap-3">
          <Button variant="primary" onClick={() => void runScan()} disabled={scan.kind === "running"} aria-busy={scan.kind === "running"}>
            {scan.kind === "running" ? "Scanning…" : "Scan Repository"}
          </Button>
          <Button onClick={() => dispatch({ type: "navigate", page: "agent" })}>Ask Agent</Button>
          <Button onClick={() => dispatch({ type: "navigate", page: "pull-requests" })}>View Pull Requests</Button>
          <Button variant="ghost" onClick={() => dispatch({ type: "clear-repository" })}>
            Choose another repository
          </Button>
        </div>
        <div role="status" className="mt-3 min-h-5 text-sm text-muted">
          {scan.kind === "done" &&
            `Scan complete: ${scan.summary.filesIndexed} files indexed, ${scan.summary.filesSkipped} skipped, ${scan.summary.chunksCreated} chunks created.`}
        </div>
        {scan.kind === "error" && (
          <div role="alert" className="mt-2 max-w-md rounded-md border border-danger/60 bg-surface px-4 py-3 text-sm">
            <p className="font-medium text-danger">The scan did not finish</p>
            <p className="mt-1 text-muted">{scan.error.message}</p>
            <Button className="mt-3" onClick={() => void runScan()}>
              Try again
            </Button>
          </div>
        )}
      </Card>
    </PageContainer>
  );
}
