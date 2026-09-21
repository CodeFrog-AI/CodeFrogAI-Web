/**
 * The file explorer's state: the tree, which folders are open, the selected file, and the
 * loading / error state of each request. A pure reducer, so it is testable without React.
 */

import type { RepositoryFileContent, RepositoryFilesError, RepositoryTree, TreeNode } from "@/lib/repository-files";

export type Load<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: RepositoryFilesError };

export interface ExplorerState {
  tree: Load<RepositoryTree>;
  /** Paths of the folders that are open. */
  expanded: readonly string[];
  selectedPath: string | null;
  file: Load<RepositoryFileContent>;
}

export type ExplorerAction =
  | { type: "tree-loading" }
  | { type: "tree-loaded"; tree: RepositoryTree }
  | { type: "tree-failed"; error: RepositoryFilesError }
  | { type: "toggle-directory"; path: string }
  | { type: "select-file"; path: string }
  | { type: "file-loaded"; file: RepositoryFileContent }
  | { type: "file-failed"; path: string; error: RepositoryFilesError };

/** The tree loads as soon as the explorer appears, so it starts in the loading state. */
export const initialExplorerState: ExplorerState = {
  tree: { status: "loading" },
  expanded: [],
  selectedPath: null,
  file: { status: "idle" },
};

export function findNode(entries: readonly TreeNode[], path: string): TreeNode | null {
  for (const entry of entries) {
    if (entry.path === path) return entry;
    if (entry.type === "directory" && path.startsWith(`${entry.path}/`)) {
      const found = findNode(entry.children, path);
      if (found) return found;
    }
  }
  return null;
}

function directoryPaths(entries: readonly TreeNode[]): string[] {
  return entries.flatMap((entry) => (entry.type === "directory" ? [entry.path, ...directoryPaths(entry.children)] : []));
}

export function explorerReducer(state: ExplorerState, action: ExplorerAction): ExplorerState {
  switch (action.type) {
    case "tree-loading":
      return { ...state, tree: { status: "loading" } };

    case "tree-loaded": {
      // A refresh keeps what still exists: open folders and the selected file.
      const folders = new Set(directoryPaths(action.tree.entries));
      const selected = state.selectedPath !== null ? findNode(action.tree.entries, state.selectedPath) : null;
      const keepSelection = selected !== null && selected.type === "file";
      return {
        tree: { status: "ready", data: action.tree },
        expanded: state.expanded.filter((path) => folders.has(path)),
        selectedPath: keepSelection ? state.selectedPath : null,
        file: keepSelection ? state.file : { status: "idle" },
      };
    }

    case "tree-failed":
      return { ...state, tree: { status: "error", error: action.error } };

    case "toggle-directory":
      return state.expanded.includes(action.path)
        ? { ...state, expanded: state.expanded.filter((path) => path !== action.path) }
        : { ...state, expanded: [...state.expanded, action.path] };

    case "select-file": {
      // Only a file that is actually in the loaded tree can be selected.
      if (state.tree.status !== "ready") return state;
      const node = findNode(state.tree.data.entries, action.path);
      if (node === null || node.type !== "file") return state;
      return { ...state, selectedPath: node.path, file: { status: "loading" } };
    }

    case "file-loaded":
      // Ignore an answer for a file that is no longer the selected one (a slower, older request).
      return action.file.path === state.selectedPath ? { ...state, file: { status: "ready", data: action.file } } : state;

    case "file-failed":
      return action.path === state.selectedPath ? { ...state, file: { status: "error", error: action.error } } : state;

    default:
      return state;
  }
}
