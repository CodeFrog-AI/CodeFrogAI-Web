/**
 * GitHub repositories through the CodeFrog backend: list the ones the signed-in user can
 * access, connect one, and scan a connected one. Uses the existing endpoints as they are:
 *
 *   GET  /api/v1/repositories/github        list, with a `connected` flag
 *   POST /api/v1/repositories/connect       { github_repository_id } -> the CodeFrog repository (UUID)
 *   POST /api/v1/repositories/{id}/scan     index the repository
 *
 * Only repository metadata is handled here: no source code, and no GitHub credentials.
 */

import { apiRequest, ApiError, type ApiDeps } from "@/lib/api";
import type { GitHubRepository } from "@/lib/app-state";

export interface GitHubRepositoryOption {
  githubRepositoryId: number;
  owner: string;
  name: string;
  defaultBranch: string;
  private: boolean;
  connected: boolean;
}

export interface ConnectedRepository {
  /** The CodeFrog repository id (a UUID) used by every other endpoint. */
  repositoryId: string;
  githubRepositoryId: number;
  owner: string;
  name: string;
  defaultBranch: string;
  private: boolean;
}

export interface ScanSummary {
  filesDiscovered: number;
  filesIndexed: number;
  filesSkipped: number;
  filesRemoved: number;
  chunksCreated: number;
}

const MAX_REPOSITORIES = 2000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MAX_GITHUB_ID = 2_147_483_647; // the backend stores GitHub ids in a 32-bit column

function invalid(): never {
  throw new ApiError("INVALID_RESPONSE");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function text(value: unknown, max = 255): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= max;
}

function count(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function githubId(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 && value <= MAX_GITHUB_ID;
}

export function parseRepositoryList(value: unknown): GitHubRepositoryOption[] {
  if (!isRecord(value) || !Array.isArray(value.repositories) || value.repositories.length > MAX_REPOSITORIES) invalid();
  return value.repositories.map((item: unknown) => {
    if (!isRecord(item)) invalid();
    const { github_repository_id: id, owner, name, default_branch: branch, private: isPrivate, connected } = item;
    if (!githubId(id) || !text(owner) || !text(name) || !text(branch)) invalid();
    if (typeof isPrivate !== "boolean" || typeof connected !== "boolean") invalid();
    return { githubRepositoryId: id, owner, name, defaultBranch: branch, private: isPrivate, connected };
  });
}

export function parseConnectedRepository(value: unknown): ConnectedRepository {
  if (!isRecord(value)) invalid();
  const { id, github_repository_id: githubRepositoryId, owner, name, default_branch: branch, private: isPrivate } = value;
  if (typeof id !== "string" || !UUID.test(id) || !githubId(githubRepositoryId)) invalid();
  if (!text(owner) || !text(name) || !text(branch) || typeof isPrivate !== "boolean") invalid();
  return { repositoryId: id, githubRepositoryId, owner, name, defaultBranch: branch, private: isPrivate };
}

export function parseScanSummary(value: unknown): ScanSummary {
  if (!isRecord(value)) invalid();
  const { files_discovered: discovered, files_indexed: indexed, files_skipped: skipped, files_removed: removed, chunks_created: chunks } = value;
  if (!count(discovered) || !count(indexed) || !count(skipped) || !count(removed) || !count(chunks)) invalid();
  return { filesDiscovered: discovered, filesIndexed: indexed, filesSkipped: skipped, filesRemoved: removed, chunksCreated: chunks };
}

export function listGitHubRepositories(deps?: ApiDeps): Promise<GitHubRepositoryOption[]> {
  return apiRequest("/api/v1/repositories/github", parseRepositoryList, {}, deps);
}

/** Connect a repository (or re-connect one already connected: the backend reuses the record). */
export function connectGitHubRepository(githubRepositoryId: number, deps?: ApiDeps): Promise<ConnectedRepository> {
  if (!githubId(githubRepositoryId)) return Promise.reject(new ApiError("BAD_REQUEST"));
  return apiRequest("/api/v1/repositories/connect", parseConnectedRepository, { method: "POST", body: { github_repository_id: githubRepositoryId } }, deps);
}

export function scanRepository(repositoryId: string, deps?: ApiDeps): Promise<ScanSummary> {
  if (!UUID.test(repositoryId)) return Promise.reject(new ApiError("BAD_REQUEST"));
  return apiRequest(`/api/v1/repositories/${repositoryId}/scan`, parseScanSummary, { method: "POST" }, deps);
}

/** Case-insensitive match on `owner/name`; an empty query matches everything. */
export function filterRepositories(repositories: readonly GitHubRepositoryOption[], query: string): GitHubRepositoryOption[] {
  const needle = query.trim().toLowerCase();
  if (needle === "") return [...repositories];
  return repositories.filter((repository) => `${repository.owner}/${repository.name}`.toLowerCase().includes(needle));
}

/** The application-state form of a connected repository. */
export function toGitHubRepository(connected: ConnectedRepository): GitHubRepository {
  return { source: "github", ...connected };
}
