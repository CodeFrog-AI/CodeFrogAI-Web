"use client";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { connectGitHub } from "@/lib/github-auth";

/** The real GitHub connection state, with Connect / Reconnect starting the OAuth flow. */
export function GitHubConnectionCard() {
  const { state, refresh } = useAuth();
  const connected = state.status === "signed-in" && state.user.githubConnected;

  return (
    <Card title="GitHub connection" description="Connect your account so CodeFrog can work with your repositories and pull requests.">
      {state.status === "loading" && (
        <p role="status" className="text-sm text-muted">
          Checking your GitHub connection…
        </p>
      )}

      {state.status === "error" && (
        <div role="alert" className="space-y-3 text-sm">
          <p className="text-muted">{(state.error instanceof ApiError ? state.error : new ApiError("SERVER_ERROR")).message}</p>
          <Button onClick={() => void refresh()}>Try again</Button>
        </div>
      )}

      {state.status === "signed-out" && (
        <div className="space-y-3 text-sm">
          {state.expired && <p className="text-muted">Your session has expired. Connect GitHub again to continue.</p>}
          <p>
            <span className="text-muted">Status: </span>
            Not connected
          </p>
          <Button variant="primary" onClick={() => connectGitHub()}>
            Connect GitHub
          </Button>
        </div>
      )}

      {state.status === "signed-in" && (
        <div className="space-y-3 text-sm">
          {connected ? (
            <>
              <p className="font-medium text-accent">✓ Connected</p>
              <p>
                <span className="text-muted">GitHub account: </span>
                <span className="font-mono">@{state.user.githubLogin}</span>
              </p>
              <Button onClick={() => connectGitHub()}>Reconnect GitHub</Button>
            </>
          ) : (
            <>
              <p>
                <span className="text-muted">Status: </span>
                Not connected
              </p>
              <Button variant="primary" onClick={() => connectGitHub()}>
                Connect GitHub
              </Button>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
