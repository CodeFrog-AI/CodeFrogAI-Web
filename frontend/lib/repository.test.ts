import { describe, expect, it } from "vitest";

import { describeBranch, describeGitStatus, describeRemote } from "@/lib/repository";

describe("describeBranch", () => {
  it("shows the branch name", () => {
    expect(describeBranch("feature/x")).toBe("feature/x");
  });

  it("explains a detached HEAD", () => {
    expect(describeBranch(null)).toBe("Detached HEAD");
  });
});

describe("describeGitStatus", () => {
  it("distinguishes clean from dirty", () => {
    expect(describeGitStatus(false)).toBe("Clean");
    expect(describeGitStatus(true)).toBe("Uncommitted changes");
  });
});

describe("describeRemote", () => {
  it("shows the remote URL", () => {
    expect(describeRemote("https://github.com/o/r.git")).toBe("https://github.com/o/r.git");
  });

  it("says when there is no remote", () => {
    expect(describeRemote(null)).toBe("No remote configured");
  });
});
