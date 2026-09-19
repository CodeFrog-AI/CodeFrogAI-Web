import type { RepositoryInfo } from "@/lib/app-state";

export const NOT_ANALYZED = "Not analyzed yet";

/**
 * Placeholder data for the UI until real repository selection exists. It is clearly a sample:
 * nothing here is read from disk or GitHub.
 */
export const SAMPLE_REPOSITORY: RepositoryInfo = {
  name: "sample-project",
  path: "~/projects/sample-project",
  branch: "main",
  status: NOT_ANALYZED,
  projectType: NOT_ANALYZED,
  languages: [],
};

export function describeLanguages(languages: readonly string[]): string {
  return languages.length > 0 ? languages.join(", ") : NOT_ANALYZED;
}
