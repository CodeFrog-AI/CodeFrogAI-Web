/**
 * Browsing the selected repository's files. Like `local-repository.ts`, this is the only place
 * that knows about Tauri: components call `listRepositoryTree()` / `readRepositoryFile()` and get
 * typed, validated data or a typed error.
 *
 * The desktop app decides which repository is browsed (the one selected with Open Repository).
 * The UI never sends a root or an absolute path, only a repository-relative file path. Nothing
 * is faked: outside the desktop app every call fails with DESKTOP_REQUIRED.
 */

import { invoke, isTauri } from "@tauri-apps/api/core";

export const LIST_TREE_COMMAND = "list_repository_tree";
export const READ_FILE_COMMAND = "read_repository_file";
export const CLEAR_SELECTION_COMMAND = "clear_selected_repository";

export type RepositoryFilesErrorCode =
  | "NO_REPOSITORY_SELECTED"
  | "INVALID_PATH"
  | "PATH_NOT_ALLOWED"
  | "PATH_OUTSIDE_REPOSITORY"
  | "NOT_FOUND"
  | "NOT_A_FILE"
  | "FILE_TOO_LARGE"
  | "NOT_UTF8_TEXT"
  | "PERMISSION_DENIED"
  | "IO_ERROR"
  | "DESKTOP_REQUIRED"
  | "INVALID_RESPONSE";

/** Fixed, user-facing messages. Raw messages from the operating system are never shown. */
export const FILES_ERROR_MESSAGES: Record<RepositoryFilesErrorCode, string> = {
  NO_REPOSITORY_SELECTED: "No repository is selected. Open a repository first.",
  INVALID_PATH: "That is not a valid path inside the repository.",
  PATH_NOT_ALLOWED: "This path is not available for browsing.",
  PATH_OUTSIDE_REPOSITORY: "This path is outside the repository.",
  NOT_FOUND: "The file was not found. It may have been moved or deleted; try refreshing.",
  NOT_A_FILE: "That path is not a file.",
  FILE_TOO_LARGE: "This file is too large to display.",
  NOT_UTF8_TEXT: "This file is not UTF-8 text (it may be binary), so it cannot be displayed.",
  PERMISSION_DENIED: "CodeFrog does not have permission to read this.",
  IO_ERROR: "The repository could not be read.",
  DESKTOP_REQUIRED: "Repository file browsing is available in the CodeFrog desktop app.",
  INVALID_RESPONSE: "The desktop app returned file information CodeFrog could not understand.",
};

const CODES = Object.keys(FILES_ERROR_MESSAGES) as RepositoryFilesErrorCode[];

export class RepositoryFilesError extends Error {
  readonly code: RepositoryFilesErrorCode;

  constructor(code: RepositoryFilesErrorCode) {
    super(FILES_ERROR_MESSAGES[code]);
    this.name = "RepositoryFilesError";
    this.code = code;
  }
}

// ------------------------------------------------------------------ types

export interface FileNode {
  type: "file";
  name: string;
  /** Repository-relative, `/`-separated. */
  path: string;
  size: number;
}

export interface DirectoryNode {
  type: "directory";
  name: string;
  path: string;
  children: TreeNode[];
}

export type TreeNode = FileNode | DirectoryNode;

export interface RepositoryTree {
  entries: TreeNode[];
  /** True when the desktop app's entry or depth limit cut the listing short. */
  truncated: boolean;
  entryCount: number;
}

export interface RepositoryFileContent {
  path: string;
  name: string;
  size: number;
  contents: string;
}

/** What the desktop app provides. Tests supply a fake; production uses `tauriFilesNative`. */
export interface FilesBridge {
  isAvailable(): boolean;
  listTree(): Promise<unknown>;
  readFile(path: string): Promise<unknown>;
  clearSelection(): Promise<void>;
}

// ------------------------------------------------------------------ validation

const NUL = String.fromCharCode(0);
const MAX_PATH_LENGTH = 1024;
const MAX_NAME_LENGTH = 255;
/** Defensive bounds when parsing a response; the desktop app enforces stricter limits itself. */
const MAX_PARSE_DEPTH = 64;
const MAX_PARSE_NODES = 20_000;
const MAX_CONTENT_LENGTH = 4_000_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasControlCharacter(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code < 32 || code === 127) return true;
  }
  return false;
}

/** A repository-relative path using `/`: no absolute, drive, backslash, `.`/`..`, or empty segments. */
export function isSafeRelativePath(path: unknown): path is string {
  if (typeof path !== "string" || path.length === 0 || path.length > MAX_PATH_LENGTH) return false;
  if (path.startsWith("/") || path.includes("\\") || path.includes(":") || hasControlCharacter(path)) return false;
  return path.split("/").every((segment) => segment !== "" && segment !== "." && segment !== "..");
}

function invalid(): never {
  throw new RepositoryFilesError("INVALID_RESPONSE");
}

function parseNode(value: unknown, parentPath: string, depth: number, counter: { nodes: number }): TreeNode {
  counter.nodes += 1;
  if (depth > MAX_PARSE_DEPTH || counter.nodes > MAX_PARSE_NODES || !isRecord(value)) invalid();
  const { name, path, type } = value;

  if (typeof name !== "string" || name === "" || name.length > MAX_NAME_LENGTH || name.includes("/") || name.includes(NUL)) invalid();
  if (!isSafeRelativePath(path)) invalid();
  // The path must be exactly the parent's path plus this node's name.
  if (path !== (parentPath === "" ? name : `${parentPath}/${name}`)) invalid();

  if (type === "file") {
    const { size } = value;
    if (typeof size !== "number" || !Number.isSafeInteger(size) || size < 0 || "children" in value) invalid();
    return { type: "file", name, path, size };
  }
  if (type === "directory") {
    const { children } = value;
    if (!Array.isArray(children) || "size" in value) invalid();
    return { type: "directory", name, path, children: children.map((child) => parseNode(child, path, depth + 1, counter)) };
  }
  return invalid();
}

/** Validate the tree command's result. Anything unexpected is an INVALID_RESPONSE, never partial data. */
export function parseRepositoryTree(value: unknown): RepositoryTree {
  if (!isRecord(value)) invalid();
  const { entries, truncated, entryCount } = value;
  if (!Array.isArray(entries) || typeof truncated !== "boolean") invalid();
  if (typeof entryCount !== "number" || !Number.isSafeInteger(entryCount) || entryCount < 0) invalid();
  const counter = { nodes: 0 };
  return { entries: entries.map((entry) => parseNode(entry, "", 0, counter)), truncated, entryCount };
}

/** Validate the file command's result. */
export function parseRepositoryFile(value: unknown): RepositoryFileContent {
  if (!isRecord(value)) invalid();
  const { path, name, size, contents } = value;
  if (!isSafeRelativePath(path) || typeof name !== "string" || name === "" || name.includes("/")) invalid();
  if (path.split("/").pop() !== name) invalid();
  if (typeof size !== "number" || !Number.isSafeInteger(size) || size < 0) invalid();
  if (typeof contents !== "string" || contents.length > MAX_CONTENT_LENGTH) invalid();
  return { path, name, size, contents };
}

/** Turn whatever a rejected command call carried into a typed error with a fixed message. */
export function toFilesError(error: unknown): RepositoryFilesError {
  if (error instanceof RepositoryFilesError) return error;
  if (isRecord(error) && typeof error.code === "string" && (CODES as string[]).includes(error.code)) {
    return new RepositoryFilesError(error.code as RepositoryFilesErrorCode);
  }
  return new RepositoryFilesError("IO_ERROR");
}

// ------------------------------------------------------------------ operations

export async function listRepositoryTree(native: FilesBridge = tauriFilesNative): Promise<RepositoryTree> {
  if (!native.isAvailable()) throw new RepositoryFilesError("DESKTOP_REQUIRED");
  let raw: unknown;
  try {
    raw = await native.listTree();
  } catch (error) {
    throw toFilesError(error);
  }
  return parseRepositoryTree(raw);
}

/** Read one file by its repository-relative path. Unsafe paths are refused before the desktop app is called. */
export async function readRepositoryFile(path: string, native: FilesBridge = tauriFilesNative): Promise<RepositoryFileContent> {
  if (!native.isAvailable()) throw new RepositoryFilesError("DESKTOP_REQUIRED");
  if (!isSafeRelativePath(path)) throw new RepositoryFilesError("INVALID_PATH");
  let raw: unknown;
  try {
    raw = await native.readFile(path);
  } catch (error) {
    throw toFilesError(error);
  }
  const file = parseRepositoryFile(raw);
  if (file.path !== path) invalid(); // the answer must be for the file that was asked for
  return file;
}

/** Tell the desktop app to forget the selected repository. A no-op outside the desktop app; never throws. */
export async function clearRepositorySelection(native: FilesBridge = tauriFilesNative): Promise<void> {
  if (!native.isAvailable()) return;
  try {
    await native.clearSelection();
  } catch {
    // Forgetting the selection is best effort; the UI has already moved on.
  }
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** The real bridge: the fixed desktop commands, and nothing else. */
export const tauriFilesNative: FilesBridge = {
  isAvailable() {
    return isTauri();
  },
  listTree() {
    return invoke(LIST_TREE_COMMAND);
  },
  readFile(path: string) {
    return invoke(READ_FILE_COMMAND, { path });
  },
  async clearSelection() {
    await invoke(CLEAR_SELECTION_COMMAND);
  },
};
