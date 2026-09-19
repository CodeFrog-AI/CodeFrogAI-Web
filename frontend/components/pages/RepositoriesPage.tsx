"use client";

import { useState } from "react";

import { PageContainer } from "@/components/layout/PageContainer";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { useApp } from "@/lib/app-context";
import type { RepositoryInfo } from "@/lib/app-state";
import { describeLanguages, SAMPLE_REPOSITORY } from "@/lib/repository";

export function RepositoriesPage() {
  const { state, dispatch } = useApp();
  return state.repository ? (
    <RepositoryDashboard repository={state.repository} />
  ) : (
    <Welcome
      onOpenRepository={() => dispatch({ type: "select-repository", repository: SAMPLE_REPOSITORY })}
      onConnectGitHub={() => dispatch({ type: "navigate", page: "settings" })}
    />
  );
}

function Welcome({ onOpenRepository, onConnectGitHub }: { onOpenRepository: () => void; onConnectGitHub: () => void }) {
  return (
    <PageContainer
      title="Welcome to CodeFrog"
      description="A local-first AI software engineer. Understand a codebase, plan changes, review diffs, and ship pull requests, with you approving every step."
    >
      <EmptyState
        title="No repository selected"
        description="Open a local repository or connect GitHub to get started. Repository selection is not connected yet, so opening a repository loads sample data."
      >
        <Button variant="primary" onClick={onOpenRepository}>
          Open Repository
        </Button>
        <Button onClick={onConnectGitHub}>Connect GitHub</Button>
      </EmptyState>
    </PageContainer>
  );
}

function RepositoryDashboard({ repository }: { repository: RepositoryInfo }) {
  const { dispatch } = useApp();
  const [notice, setNotice] = useState("");

  return (
    <PageContainer title={repository.name} description="Repository dashboard (sample data)">
      <Card title="Repository">
        <dl className="grid gap-4 text-sm sm:grid-cols-2">
          <Detail label="Name" value={repository.name} />
          <Detail label="Path" value={repository.path} mono />
          <Detail label="Current branch" value={repository.branch} mono />
          <Detail label="Status" value={repository.status} />
          <Detail label="Project type" value={repository.projectType} />
          <Detail label="Languages" value={describeLanguages(repository.languages)} />
        </dl>
      </Card>

      <Card title="Quick actions">
        <div className="flex flex-wrap gap-3">
          <Button variant="primary" onClick={() => dispatch({ type: "navigate", page: "agent" })}>
            Ask Agent
          </Button>
          <Button onClick={() => setNotice("Repository analysis is not connected yet.")}>Analyze Repository</Button>
          <Button onClick={() => dispatch({ type: "navigate", page: "pull-requests" })}>View Pull Requests</Button>
          <Button variant="ghost" onClick={() => dispatch({ type: "clear-repository" })}>
            Close repository
          </Button>
        </div>
        <p role="status" className="mt-3 min-h-5 text-sm text-muted">
          {notice}
        </p>
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
