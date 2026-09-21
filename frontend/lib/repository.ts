import type { LocalRepository } from "@/lib/app-state";

/** Display helpers for a selected repository. */

export function describeBranch(branch: LocalRepository["branch"]): string {
  return branch ?? "Detached HEAD";
}

export function describeGitStatus(isDirty: boolean): string {
  return isDirty ? "Uncommitted changes" : "Clean";
}

export function describeRemote(remoteUrl: LocalRepository["remoteUrl"]): string {
  return remoteUrl ?? "No remote configured";
}
