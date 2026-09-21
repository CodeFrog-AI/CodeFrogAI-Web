import type { RepositoryInfo } from "@/lib/app-state";

/** A repository shaped like the desktop app's real answer, for tests only. */
export const TEST_REPOSITORY: RepositoryInfo = {
  name: "codefrog",
  path: "/home/dev/projects/codefrog",
  branch: "main",
  isDirty: false,
  remoteUrl: "https://github.com/example/codefrog.git",
};
