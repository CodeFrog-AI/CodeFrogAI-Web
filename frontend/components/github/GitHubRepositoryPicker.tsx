"use client";

import { useEffect, useId, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { ApiError } from "@/lib/api";
import { useApp } from "@/lib/app-context";
import { useAuth } from "@/lib/auth-context";
import { connectGitHub } from "@/lib/github-auth";
import {
  connectGitHubRepository,
  filterRepositories,
  listGitHubRepositories,
  toGitHubRepository,
  type GitHubRepositoryOption,
} from "@/lib/github-repositories";

type Load = { status: "loading" } | { status: "ready"; repositories: GitHubRepositoryOption[] } | { status: "error"; error: ApiError };

function asApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError("SERVER_ERROR");
}

/** Pick one of your GitHub repositories and connect it to CodeFrog. */
export function GitHubRepositoryPicker() {
  const { dispatch } = useApp();
  const auth = useAuth();
  const searchId = useId();
  const [load, setLoad] = useState<Load>({ status: "loading" });
  const [query, setQuery] = useState("");
  const [connecting, setConnecting] = useState<number | null>(null);
  const [connectError, setConnectError] = useState<ApiError | null>(null);
  const [attempt, setAttempt] = useState(0);
  const { expire } = auth;

  useEffect(() => {
    let cancelled = false;
    listGitHubRepositories().then(
      (repositories) => {
        if (!cancelled) setLoad({ status: "ready", repositories });
      },
      (error: unknown) => {
        if (cancelled) return;
        const apiError = asApiError(error);
        if (apiError.code === "UNAUTHORIZED") expire(); // the session is gone: back to Connect GitHub
        else setLoad({ status: "error", error: apiError });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [attempt, expire]);

  function retry() {
    setLoad({ status: "loading" });
    setAttempt((value) => value + 1);
  }

  async function connect(repository: GitHubRepositoryOption) {
    setConnecting(repository.githubRepositoryId);
    setConnectError(null);
    try {
      const connected = await connectGitHubRepository(repository.githubRepositoryId);
      dispatch({ type: "select-repository", repository: toGitHubRepository(connected) });
    } catch (error) {
      const apiError = asApiError(error);
      if (apiError.code === "UNAUTHORIZED") auth.expire();
      else {
        setConnectError(apiError);
        setConnecting(null);
      }
    }
  }

  return (
    <Card title="GitHub repositories" description="Choose the repository CodeFrog should work with.">
      {load.status === "loading" && (
        <p role="status" className="text-sm text-muted">
          Loading your repositories…
        </p>
      )}

      {load.status === "error" && (
        <div role="alert" className="space-y-3 text-sm">
          <p className="font-medium text-danger">Could not load your repositories</p>
          <p className="text-muted">{load.error.message}</p>
          <div className="flex flex-wrap gap-3">
            {load.error.code === "FORBIDDEN" ? (
              <Button variant="primary" onClick={() => connectGitHub()}>
                Reconnect GitHub
              </Button>
            ) : (
              <Button onClick={retry}>Try again</Button>
            )}
          </div>
        </div>
      )}

      {load.status === "ready" && (
        <RepositoryList
          repositories={load.repositories}
          query={query}
          searchId={searchId}
          connecting={connecting}
          connectError={connectError}
          onQuery={setQuery}
          onConnect={(repository) => void connect(repository)}
          onReconnect={() => connectGitHub()}
        />
      )}
    </Card>
  );
}

function RepositoryList({
  repositories,
  query,
  searchId,
  connecting,
  connectError,
  onQuery,
  onConnect,
  onReconnect,
}: {
  repositories: GitHubRepositoryOption[];
  query: string;
  searchId: string;
  connecting: number | null;
  connectError: ApiError | null;
  onQuery: (query: string) => void;
  onConnect: (repository: GitHubRepositoryOption) => void;
  onReconnect: () => void;
}) {
  if (repositories.length === 0) {
    return (
      <EmptyState
        icon="repositories"
        title="No repositories found"
        description="Your GitHub account has no repositories CodeFrog can see. If you expected some, reconnect GitHub and allow repository access."
      >
        <Button onClick={onReconnect}>Reconnect GitHub</Button>
      </EmptyState>
    );
  }

  const visible = filterRepositories(repositories, query);
  return (
    <div>
      <label htmlFor={searchId} className="block text-sm font-medium">
        Search repositories
      </label>
      <input
        id={searchId}
        type="search"
        value={query}
        onChange={(event) => onQuery(event.target.value)}
        placeholder="Search repositories…"
        autoComplete="off"
        className="mt-1 w-full rounded-md border border-border bg-background px-3 py-2 text-sm placeholder:text-muted"
      />

      {connectError && (
        <div role="alert" className="mt-3 rounded-md border border-danger/60 px-3 py-2 text-sm">
          <p className="font-medium text-danger">Could not connect the repository</p>
          <p className="mt-1 text-muted">{connectError.message}</p>
          {connectError.code === "FORBIDDEN" && (
            <Button className="mt-2" onClick={onReconnect}>
              Reconnect GitHub
            </Button>
          )}
        </div>
      )}

      {visible.length === 0 ? (
        <p role="status" className="mt-4 text-sm text-muted">
          No repositories match &ldquo;{query.trim()}&rdquo;.
        </p>
      ) : (
        <ul aria-label="GitHub repositories" className="mt-4 max-h-[28rem] divide-y divide-border overflow-auto rounded-md border border-border">
          {visible.map((repository) => {
            const busy = connecting === repository.githubRepositoryId;
            return (
              <li key={repository.githubRepositoryId} className="flex items-center justify-between gap-4 px-4 py-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-semibold">{repository.name}</p>
                  <p className="truncate font-mono text-xs text-muted">
                    {repository.owner} / {repository.name}
                  </p>
                  <p className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted">
                    <span className="font-mono">{repository.defaultBranch}</span>
                    <span aria-hidden="true">·</span>
                    <span>{repository.private ? "Private" : "Public"}</span>
                    <span aria-hidden="true">·</span>
                    <span className={repository.connected ? "text-accent" : ""}>{repository.connected ? "Connected" : "Not connected"}</span>
                  </p>
                </div>
                <Button
                  variant={repository.connected ? "secondary" : "primary"}
                  disabled={connecting !== null}
                  aria-busy={busy}
                  aria-label={`${repository.connected ? "Select" : "Connect"} ${repository.owner}/${repository.name}`}
                  onClick={() => onConnect(repository)}
                >
                  {busy ? "Connecting…" : repository.connected ? "Select" : "Connect"}
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
