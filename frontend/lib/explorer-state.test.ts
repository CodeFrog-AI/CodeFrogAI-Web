import { describe, expect, it } from "vitest";

import { explorerReducer, findNode, initialExplorerState, type ExplorerAction, type ExplorerState } from "@/lib/explorer-state";
import { parseRepositoryTree, RepositoryFilesError, type RepositoryFileContent, type RepositoryTree } from "@/lib/repository-files";

const TREE: RepositoryTree = parseRepositoryTree({
  entries: [
    {
      name: "src",
      path: "src",
      type: "directory",
      children: [
        { name: "app", path: "src/app", type: "directory", children: [{ name: "page.tsx", path: "src/app/page.tsx", type: "file", size: 10 }] },
        { name: "index.ts", path: "src/index.ts", type: "file", size: 5 },
      ],
    },
    { name: "README.md", path: "README.md", type: "file", size: 7 },
  ],
  truncated: false,
  entryCount: 5,
});

const FILE: RepositoryFileContent = { path: "src/index.ts", name: "index.ts", size: 5, contents: "hello" };

function run(actions: ExplorerAction[], from: ExplorerState = initialExplorerState): ExplorerState {
  return actions.reduce(explorerReducer, from);
}

const loaded = run([{ type: "tree-loaded", tree: TREE }]);

describe("loading the tree", () => {
  it("starts in the loading state with nothing selected", () => {
    expect(initialExplorerState.tree).toEqual({ status: "loading" });
    expect(initialExplorerState.selectedPath).toBeNull();
    expect(initialExplorerState.file).toEqual({ status: "idle" });
  });

  it("holds the tree once loaded", () => {
    expect(loaded.tree).toEqual({ status: "ready", data: TREE });
  });

  it("records an error, and the error can be retried by loading again", () => {
    const error = new RepositoryFilesError("IO_ERROR");
    const failed = run([{ type: "tree-failed", error }]);
    expect(failed.tree).toEqual({ status: "error", error });
    expect(run([{ type: "tree-loading" }], failed).tree).toEqual({ status: "loading" });
  });

  it("carries the desktop-required error for browser mode", () => {
    const failed = run([{ type: "tree-failed", error: new RepositoryFilesError("DESKTOP_REQUIRED") }]);
    expect(failed.tree.status === "error" && failed.tree.error.message).toBe("Repository file browsing is available in the CodeFrog desktop app.");
  });
});

describe("expanding folders", () => {
  it("toggles a folder open and closed", () => {
    const open = run([{ type: "toggle-directory", path: "src" }], loaded);
    expect(open.expanded).toEqual(["src"]);
    expect(run([{ type: "toggle-directory", path: "src" }], open).expanded).toEqual([]);
  });

  it("keeps folders independent", () => {
    const state = run([{ type: "toggle-directory", path: "src" }, { type: "toggle-directory", path: "src/app" }, { type: "toggle-directory", path: "src" }], loaded);
    expect(state.expanded).toEqual(["src/app"]);
  });
});

describe("selecting a file", () => {
  it("selects a file in the tree and starts loading it", () => {
    const state = run([{ type: "select-file", path: "src/index.ts" }], loaded);
    expect(state.selectedPath).toBe("src/index.ts");
    expect(state.file).toEqual({ status: "loading" });
  });

  it("shows the file once it loads", () => {
    const state = run([{ type: "select-file", path: "src/index.ts" }, { type: "file-loaded", file: FILE }], loaded);
    expect(state.file).toEqual({ status: "ready", data: FILE });
  });

  it("records a read error for the selected file", () => {
    const error = new RepositoryFilesError("FILE_TOO_LARGE");
    const state = run([{ type: "select-file", path: "src/index.ts" }, { type: "file-failed", path: "src/index.ts", error }], loaded);
    expect(state.file).toEqual({ status: "error", error });
  });

  it("ignores selecting a folder, a missing path, or anything before the tree has loaded", () => {
    expect(run([{ type: "select-file", path: "src" }], loaded)).toBe(loaded);
    expect(run([{ type: "select-file", path: "nope.ts" }], loaded)).toBe(loaded);
    expect(run([{ type: "select-file", path: "../../.env" }], loaded)).toBe(loaded);
    expect(run([{ type: "select-file", path: "README.md" }])).toBe(initialExplorerState);
  });

  it("ignores a late answer for a file that is no longer selected", () => {
    const state = run(
      [
        { type: "select-file", path: "src/index.ts" },
        { type: "select-file", path: "README.md" },
        { type: "file-loaded", file: FILE },
        { type: "file-failed", path: "src/index.ts", error: new RepositoryFilesError("IO_ERROR") },
      ],
      loaded,
    );
    expect(state.selectedPath).toBe("README.md");
    expect(state.file).toEqual({ status: "loading" });
  });

  it("can select the same file again to retry after an error", () => {
    const failed = run([{ type: "select-file", path: "src/index.ts" }, { type: "file-failed", path: "src/index.ts", error: new RepositoryFilesError("IO_ERROR") }], loaded);
    expect(run([{ type: "select-file", path: "src/index.ts" }], failed).file).toEqual({ status: "loading" });
  });
});

describe("refreshing", () => {
  const withSelection = run([{ type: "toggle-directory", path: "src" }, { type: "toggle-directory", path: "src/app" }, { type: "select-file", path: "src/index.ts" }, { type: "file-loaded", file: FILE }], loaded);

  it("keeps open folders and the selected file that still exist", () => {
    const refreshed = run([{ type: "tree-loading" }, { type: "tree-loaded", tree: TREE }], withSelection);
    expect(refreshed.expanded).toEqual(["src", "src/app"]);
    expect(refreshed.selectedPath).toBe("src/index.ts");
    expect(refreshed.file).toEqual({ status: "ready", data: FILE });
  });

  it("drops open folders and the selection that no longer exist", () => {
    const smaller: RepositoryTree = parseRepositoryTree({
      entries: [{ name: "README.md", path: "README.md", type: "file", size: 7 }],
      truncated: false,
      entryCount: 1,
    });
    const refreshed = run([{ type: "tree-loaded", tree: smaller }], withSelection);
    expect(refreshed.expanded).toEqual([]);
    expect(refreshed.selectedPath).toBeNull();
    expect(refreshed.file).toEqual({ status: "idle" });
  });

  it("drops a selection that became a folder", () => {
    const changed: RepositoryTree = parseRepositoryTree({
      entries: [{ name: "src", path: "src", type: "directory", children: [{ name: "index.ts", path: "src/index.ts", type: "directory", children: [] }] }],
      truncated: false,
      entryCount: 2,
    });
    expect(run([{ type: "tree-loaded", tree: changed }], withSelection).selectedPath).toBeNull();
  });

  it("does not mutate the previous state", () => {
    const before = structuredClone(withSelection.expanded);
    run([{ type: "toggle-directory", path: "src" }], withSelection);
    expect(withSelection.expanded).toEqual(before);
  });
});

describe("findNode", () => {
  it("finds files and folders by path", () => {
    expect(findNode(TREE.entries, "src/app/page.tsx")?.type).toBe("file");
    expect(findNode(TREE.entries, "src/app")?.type).toBe("directory");
    expect(findNode(TREE.entries, "README.md")?.name).toBe("README.md");
  });

  it("returns null for anything else", () => {
    expect(findNode(TREE.entries, "src/missing.ts")).toBeNull();
    expect(findNode(TREE.entries, "../../.env")).toBeNull();
    expect(findNode(TREE.entries, "srcx/index.ts")).toBeNull();
  });
});
