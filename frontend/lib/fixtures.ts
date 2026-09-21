import type { GitHubRepository, LocalRepository } from "@/lib/app-state";

/** A repository shaped like the desktop app's real answer, for tests only. */
export const TEST_REPOSITORY: LocalRepository = {
  source: "local",
  name: "codefrog",
  path: "/home/dev/projects/codefrog",
  branch: "main",
  isDirty: false,
  remoteUrl: "https://github.com/example/codefrog.git",
};

/** A connected GitHub repository, for tests only. */
export const TEST_GITHUB_REPOSITORY: GitHubRepository = {
  source: "github",
  repositoryId: "3f2b8c1e-5d4a-4c6b-9e7f-1a2b3c4d5e6f",
  githubRepositoryId: 123456,
  owner: "octocat",
  name: "hello-world",
  defaultBranch: "main",
  private: false,
};
