"use client";

import { useEffect, useReducer, useRef } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { explorerReducer, initialExplorerState, type ExplorerState } from "@/lib/explorer-state";
import {
  formatFileSize,
  listRepositoryTree,
  readRepositoryFile,
  toFilesError,
  type RepositoryFilesError,
  type TreeNode,
} from "@/lib/repository-files";

/** Errors that a retry cannot fix. */
const NO_RETRY = new Set(["DESKTOP_REQUIRED", "FILE_TOO_LARGE", "NOT_UTF8_TEXT", "PATH_NOT_ALLOWED"]);

/**
 * The file tree and file viewer of the selected repository. It is remounted (keyed by the
 * repository path) when another repository is opened, so no state leaks between repositories.
 * Reading only: nothing here changes the repository.
 */
export function RepositoryExplorer() {
  const [state, dispatch] = useReducer(explorerReducer, initialExplorerState);
  const treeRequest = useRef(0);
  const stateRef = useRef<ExplorerState>(state);

  useEffect(() => {
    stateRef.current = state;
  });

  async function fetchTree() {
    const request = ++treeRequest.current;
    try {
      const tree = await listRepositoryTree();
      if (request !== treeRequest.current) return;
      dispatch({ type: "tree-loaded", tree });
      // A refresh re-reads the open file too, so the viewer is not left showing stale text.
      const selected = stateRef.current.selectedPath;
      if (selected !== null && stateRef.current.file.status === "ready") void fetchFile(selected);
    } catch (error) {
      if (request === treeRequest.current) dispatch({ type: "tree-failed", error: toFilesError(error) });
    }
  }

  async function fetchFile(path: string) {
    try {
      const file = await readRepositoryFile(path);
      dispatch({ type: "file-loaded", file });
    } catch (error) {
      dispatch({ type: "file-failed", path, error: toFilesError(error) });
    }
  }

  useEffect(() => {
    void fetchTree();
    // Load once when the explorer appears; the initial state is already "loading".
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function refresh() {
    dispatch({ type: "tree-loading" });
    void fetchTree();
  }

  function openFile(path: string) {
    dispatch({ type: "select-file", path });
    void fetchFile(path);
  }

  const loading = state.tree.status === "loading";

  return (
    <Card title="Files">
      <div className="mb-4 flex items-center justify-between gap-3">
        <p className="text-sm text-muted">Read-only view of the repository&apos;s files.</p>
        <Button onClick={refresh} disabled={loading} aria-busy={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <div className="min-w-0 rounded-md border border-border bg-background p-2">
          <TreePanel state={state} onToggle={(path) => dispatch({ type: "toggle-directory", path })} onOpenFile={openFile} onRetry={refresh} />
        </div>
        <div className="min-w-0 rounded-md border border-border bg-background">
          <FileViewer state={state} onRetry={() => state.selectedPath && openFile(state.selectedPath)} />
        </div>
      </div>
    </Card>
  );
}

function ErrorMessage({ title, error, onRetry }: { title: string; error: RepositoryFilesError; onRetry: () => void }) {
  return (
    <div role="alert" className="p-3 text-sm">
      <p className="font-medium text-danger">{title}</p>
      <p className="mt-1 text-muted">{error.message}</p>
      {!NO_RETRY.has(error.code) && (
        <Button className="mt-3" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

function TreePanel({
  state,
  onToggle,
  onOpenFile,
  onRetry,
}: {
  state: ExplorerState;
  onToggle: (path: string) => void;
  onOpenFile: (path: string) => void;
  onRetry: () => void;
}) {
  const { tree } = state;
  if (tree.status === "loading" || tree.status === "idle") {
    return (
      <p role="status" className="p-3 text-sm text-muted">
        Loading files…
      </p>
    );
  }
  if (tree.status === "error") {
    return <ErrorMessage title="Could not load the files" error={tree.error} onRetry={onRetry} />;
  }
  if (tree.data.entries.length === 0) {
    return <EmptyState icon="repositories" title="No files to show" description="This repository has no files outside the ignored folders (.git, node_modules, build output)." />;
  }
  return (
    <div className="max-h-[28rem] overflow-auto">
      <TreeList nodes={tree.data.entries} depth={0} state={state} onToggle={onToggle} onOpenFile={onOpenFile} label="Repository files" />
      {tree.data.truncated && (
        <p className="p-2 text-xs text-muted">
          Large repository: showing the first {tree.data.entryCount} entries. Some folders or files are not listed.
        </p>
      )}
    </div>
  );
}

function TreeList({
  nodes,
  depth,
  state,
  onToggle,
  onOpenFile,
  label,
}: {
  nodes: readonly TreeNode[];
  depth: number;
  state: ExplorerState;
  onToggle: (path: string) => void;
  onOpenFile: (path: string) => void;
  label?: string;
}) {
  return (
    <ul aria-label={label} className="text-sm">
      {nodes.map((node) => {
        const indent = { paddingLeft: `${depth * 14 + 8}px` };
        if (node.type === "directory") {
          const open = state.expanded.includes(node.path);
          return (
            <li key={node.path}>
              <button
                type="button"
                onClick={() => onToggle(node.path)}
                aria-expanded={open}
                style={indent}
                className="flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-surface-raised"
              >
                <span aria-hidden="true" className="w-3 text-muted">
                  {open ? "▾" : "▸"}
                </span>
                <span className="truncate">{node.name}</span>
              </button>
              {open && node.children.length > 0 && (
                <TreeList nodes={node.children} depth={depth + 1} state={state} onToggle={onToggle} onOpenFile={onOpenFile} label={`${node.name} contents`} />
              )}
            </li>
          );
        }
        const selected = state.selectedPath === node.path;
        return (
          <li key={node.path}>
            <button
              type="button"
              onClick={() => onOpenFile(node.path)}
              aria-current={selected ? "true" : undefined}
              style={indent}
              className={`flex w-full items-center gap-2 rounded px-2 py-1 text-left ${selected ? "bg-surface-raised text-accent" : "hover:bg-surface-raised"}`}
            >
              <span aria-hidden="true" className="w-3" />
              <span className="truncate">{node.name}</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function FileViewer({ state, onRetry }: { state: ExplorerState; onRetry: () => void }) {
  const { file } = state;
  if (state.selectedPath === null || file.status === "idle") {
    return <p className="p-4 text-sm text-muted">Select a file to view its contents.</p>;
  }
  if (file.status === "loading") {
    return (
      <p role="status" className="p-4 text-sm text-muted">
        Loading {state.selectedPath}…
      </p>
    );
  }
  if (file.status === "error") {
    return <ErrorMessage title="Could not open the file" error={file.error} onRetry={onRetry} />;
  }
  const { name, path, size, contents } = file.data;
  return (
    <div>
      <div className="border-b border-border px-4 py-3">
        <h3 className="truncate text-sm font-semibold">{name}</h3>
        <p className="mt-0.5 break-all font-mono text-xs text-muted">{path}</p>
        <p className="mt-0.5 text-xs text-muted">{formatFileSize(size)}</p>
      </div>
      {/* The contents are text: React escapes them, and whitespace and line breaks are preserved. */}
      <pre tabIndex={0} aria-label={`Contents of ${name}`} className="max-h-[28rem] overflow-auto whitespace-pre p-4 font-mono text-xs leading-relaxed">
        {contents === "" ? "(empty file)" : contents}
      </pre>
    </div>
  );
}
