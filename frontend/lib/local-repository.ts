/**
 * Selecting a local repository. This is the only module that knows about Tauri: React calls
 * `selectLocalRepository()` and gets typed data or a typed error back.
 *
 * Flow: native folder picker -> the `select_repository` command (validates the folder and
 * reads Git metadata) -> a parsed, validated `RepositoryInfo`. Nothing is faked: outside the
 * desktop app the call fails with DESKTOP_REQUIRED.
 */

import { invoke, isTauri } from "@tauri-apps/api/core";

import type { RepositoryInfo } from "@/lib/app-state";

export const SELECT_REPOSITORY_COMMAND = "select_repository";

export type RepositoryErrorCode =
  | "INVALID_PATH"
  | "NOT_A_GIT_REPOSITORY"
  | "PERMISSION_DENIED"
  | "GIT_UNAVAILABLE"
  | "GIT_ERROR"
  | "DESKTOP_REQUIRED"
  | "PICKER_FAILED"
  | "INVALID_RESPONSE";

/** Fixed, user-facing messages. Raw messages from the operating system or Git are never shown. */
export const ERROR_MESSAGES: Record<RepositoryErrorCode, string> = {
  INVALID_PATH: "The selected path is not an accessible folder.",
  NOT_A_GIT_REPOSITORY: "This folder is not a Git repository. Choose the folder that contains a .git directory.",
  PERMISSION_DENIED: "CodeFrog does not have permission to read this folder.",
  GIT_UNAVAILABLE: "Git is not installed or could not be started. Install Git and try again.",
  GIT_ERROR: "Git could not read this repository.",
  DESKTOP_REQUIRED: "Open Repository is available in the CodeFrog desktop app.",
  PICKER_FAILED: "The folder picker could not be opened.",
  INVALID_RESPONSE: "The desktop app returned repository information CodeFrog could not understand.",
};

const CODES = Object.keys(ERROR_MESSAGES) as RepositoryErrorCode[];
const NUL = String.fromCharCode(0);
const MAX_PATH_LENGTH = 4096;
const MAX_TEXT_LENGTH = 2048;

export class LocalRepositoryError extends Error {
  readonly code: RepositoryErrorCode;

  constructor(code: RepositoryErrorCode) {
    super(ERROR_MESSAGES[code]);
    this.name = "LocalRepositoryError";
    this.code = code;
  }
}

export type SelectionResult = { status: "selected"; repository: RepositoryInfo } | { status: "cancelled" };

/** What the desktop app provides. Tests supply a fake; production uses `tauriNative`. */
export interface NativeBridge {
  isAvailable(): boolean;
  /** The chosen folder, or null if the user cancelled. */
  pickDirectory(): Promise<string | null>;
  /** Invokes the `select_repository` command. Resolves with its raw result or rejects with its raw error. */
  inspectRepository(path: string): Promise<unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isText(value: unknown, max: number): value is string {
  return typeof value === "string" && value.trim().length > 0 && value.length <= max && !value.includes(NUL);
}

/** Remove `user:password@` from a URL, in case a credential ever gets this far. */
export function stripUrlCredentials(url: string): string {
  return url.replace(/^([a-z][a-z0-9+.-]*:\/\/)[^/@\s]*@/i, "$1");
}

/** Validate the command's result. Anything unexpected is an INVALID_RESPONSE, never partial data. */
export function parseRepositoryInfo(value: unknown): RepositoryInfo {
  if (!isRecord(value)) throw new LocalRepositoryError("INVALID_RESPONSE");
  const { name, path, branch, isDirty, remoteUrl } = value;

  if (!isText(name, MAX_TEXT_LENGTH) || !isText(path, MAX_PATH_LENGTH)) {
    throw new LocalRepositoryError("INVALID_RESPONSE");
  }
  if (branch !== null && !isText(branch, MAX_TEXT_LENGTH)) throw new LocalRepositoryError("INVALID_RESPONSE");
  if (typeof isDirty !== "boolean") throw new LocalRepositoryError("INVALID_RESPONSE");
  if (remoteUrl !== null && !isText(remoteUrl, MAX_TEXT_LENGTH)) throw new LocalRepositoryError("INVALID_RESPONSE");

  return {
    name,
    path,
    branch,
    isDirty,
    remoteUrl: remoteUrl === null ? null : stripUrlCredentials(remoteUrl),
  };
}

/** Turn whatever a rejected command call carried into a typed error with a fixed message. */
export function toRepositoryError(error: unknown): LocalRepositoryError {
  if (error instanceof LocalRepositoryError) return error;
  if (isRecord(error) && typeof error.code === "string" && (CODES as string[]).includes(error.code)) {
    return new LocalRepositoryError(error.code as RepositoryErrorCode);
  }
  return new LocalRepositoryError("GIT_ERROR");
}

export async function selectLocalRepository(native: NativeBridge = tauriNative): Promise<SelectionResult> {
  if (!native.isAvailable()) throw new LocalRepositoryError("DESKTOP_REQUIRED");

  let path: string | null;
  try {
    path = await native.pickDirectory();
  } catch {
    throw new LocalRepositoryError("PICKER_FAILED");
  }
  if (path === null) return { status: "cancelled" };
  if (typeof path !== "string" || path.trim() === "") throw new LocalRepositoryError("INVALID_PATH");

  let raw: unknown;
  try {
    raw = await native.inspectRepository(path);
  } catch (error) {
    throw toRepositoryError(error);
  }
  return { status: "selected", repository: parseRepositoryInfo(raw) };
}

/** The real bridge. The folder-picker plugin is loaded on demand, only inside the desktop app. */
export const tauriNative: NativeBridge = {
  isAvailable() {
    return isTauri();
  },
  async pickDirectory() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({ directory: true, multiple: false, title: "Open Repository" });
    return typeof selected === "string" ? selected : null;
  },
  async inspectRepository(path: string) {
    return invoke(SELECT_REPOSITORY_COMMAND, { path });
  },
};
