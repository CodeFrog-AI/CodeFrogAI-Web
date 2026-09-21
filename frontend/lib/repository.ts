import type { RepositoryInfo } from "@/lib/app-state";

/** Display helpers for a selected repository. */

export function describeBranch(branch: RepositoryInfo["branch"]): string {
  return branch ?? "Detached HEAD";
}

export function describeGitStatus(isDirty: boolean): string {
  return isDirty ? "Uncommitted changes" : "Clean";
}

export function describeRemote(remoteUrl: RepositoryInfo["remoteUrl"]): string {
  return remoteUrl ?? "No remote configured";
}
