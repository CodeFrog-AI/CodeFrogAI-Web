import type { SelectedRepository } from "@/lib/app-state";
import { describeBranch } from "@/lib/repository";

interface HeaderProps {
  title: string;
  repository: SelectedRepository | null;
}

export function Header({ title, repository }: HeaderProps) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-surface px-6">
      <p className="truncate text-sm font-medium">{title}</p>
      <p className="truncate text-sm text-muted">
        {repository ? (
          <>
            <span className="sr-only">Selected repository: </span>
            <span className="font-mono text-foreground">{repository.name}</span>
            <span aria-hidden="true"> · </span>
            <span className="font-mono">{repository.source === "local" ? describeBranch(repository.branch) : repository.defaultBranch}</span>
          </>
        ) : (
          "No repository selected"
        )}
      </p>
    </header>
  );
}
