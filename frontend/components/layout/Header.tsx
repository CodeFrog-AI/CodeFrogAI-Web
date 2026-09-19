import type { RepositoryInfo } from "@/lib/app-state";

interface HeaderProps {
  title: string;
  repository: RepositoryInfo | null;
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
            <span className="font-mono">{repository.branch}</span>
          </>
        ) : (
          "No repository selected"
        )}
      </p>
    </header>
  );
}
