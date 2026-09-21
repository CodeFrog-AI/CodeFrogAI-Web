import { describe, expect, it, vi } from "vitest";

import { TEST_REPOSITORY } from "@/lib/fixtures";
import {
  ERROR_MESSAGES,
  LocalRepositoryError,
  parseRepositoryInfo,
  selectLocalRepository,
  SELECT_REPOSITORY_COMMAND,
  stripUrlCredentials,
  toRepositoryError,
  type NativeBridge,
  type RepositoryErrorCode,
} from "@/lib/local-repository";

const NUL = String.fromCharCode(0);
const VALID_RESPONSE = { ...TEST_REPOSITORY };

function bridge(overrides: Partial<NativeBridge> = {}): NativeBridge {
  return {
    isAvailable: () => true,
    pickDirectory: async () => TEST_REPOSITORY.path,
    inspectRepository: async () => VALID_RESPONSE,
    ...overrides,
  };
}

async function failureOf(promise: Promise<unknown>): Promise<LocalRepositoryError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(LocalRepositoryError);
    return error as LocalRepositoryError;
  }
  throw new Error("expected the call to fail");
}

describe("parseRepositoryInfo", () => {
  it("accepts a valid response", () => {
    expect(parseRepositoryInfo(VALID_RESPONSE)).toEqual(TEST_REPOSITORY);
  });

  it("accepts a detached HEAD, a dirty tree, and no remote", () => {
    const parsed = parseRepositoryInfo({ ...VALID_RESPONSE, branch: null, isDirty: true, remoteUrl: null });
    expect(parsed).toEqual({ ...TEST_REPOSITORY, branch: null, isDirty: true, remoteUrl: null });
  });

  it("copies only the known fields", () => {
    const parsed = parseRepositoryInfo({ ...VALID_RESPONSE, token: "secret", env: { PATH: "x" } });
    expect(Object.keys(parsed).sort()).toEqual(["branch", "isDirty", "name", "path", "remoteUrl"]);
  });

  it("removes credentials from the remote URL", () => {
    const parsed = parseRepositoryInfo({ ...VALID_RESPONSE, remoteUrl: "https://user:hunter2@github.com/o/r.git" });
    expect(parsed.remoteUrl).toBe("https://github.com/o/r.git");
  });

  it.each([
    ["null", null],
    ["a string", "repository"],
    ["an array", []],
    ["a number", 5],
    ["an empty object", {}],
    ["a missing name", { ...VALID_RESPONSE, name: undefined }],
    ["an empty name", { ...VALID_RESPONSE, name: "  " }],
    ["a numeric name", { ...VALID_RESPONSE, name: 5 }],
    ["a missing path", { ...VALID_RESPONSE, path: undefined }],
    ["an empty path", { ...VALID_RESPONSE, path: "" }],
    ["a path with a NUL character", { ...VALID_RESPONSE, path: "/a" + NUL + "b" }],
    ["an over-long path", { ...VALID_RESPONSE, path: "/" + "a".repeat(5000) }],
    ["a numeric branch", { ...VALID_RESPONSE, branch: 1 }],
    ["an undefined branch", { ...VALID_RESPONSE, branch: undefined }],
    ["a string isDirty", { ...VALID_RESPONSE, isDirty: "false" }],
    ["a missing isDirty", { ...VALID_RESPONSE, isDirty: undefined }],
    ["a numeric remote", { ...VALID_RESPONSE, remoteUrl: 3 }],
    ["an undefined remote", { ...VALID_RESPONSE, remoteUrl: undefined }],
  ])("rejects %s as an invalid response", (_label, value) => {
    expect(() => parseRepositoryInfo(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });
});

describe("stripUrlCredentials", () => {
  it.each([
    ["https://user:token@github.com/o/r.git", "https://github.com/o/r.git"],
    ["https://token@github.com/o/r.git", "https://github.com/o/r.git"],
    ["ssh://git@github.com/o/r.git", "ssh://github.com/o/r.git"],
    ["https://github.com/o/r.git", "https://github.com/o/r.git"],
    ["git@github.com:o/r.git", "git@github.com:o/r.git"],
    ["https://github.com/o/r.git?email=a@b.c", "https://github.com/o/r.git?email=a@b.c"],
  ])("%s", (input, expected) => {
    expect(stripUrlCredentials(input)).toBe(expected);
  });
});

describe("toRepositoryError", () => {
  it("keeps the code of a structured command error but never its raw message", () => {
    const error = toRepositoryError({ code: "NOT_A_GIT_REPOSITORY", message: "os error 2: C:\\secret\\path" });
    expect(error.code).toBe("NOT_A_GIT_REPOSITORY");
    expect(error.message).toBe(ERROR_MESSAGES.NOT_A_GIT_REPOSITORY);
    expect(error.message).not.toContain("secret");
  });

  it.each(["INVALID_PATH", "NOT_A_GIT_REPOSITORY", "PERMISSION_DENIED", "GIT_UNAVAILABLE", "GIT_ERROR"] satisfies RepositoryErrorCode[])(
    "recognizes %s",
    (code) => {
      expect(toRepositoryError({ code, message: "raw" }).code).toBe(code);
    },
  );

  it.each([
    ["an unknown code", { code: "SOMETHING_ELSE", message: "raw" }],
    ["a plain string", "raw failure text with C:\\Users\\me"],
    ["an Error", new Error("boom")],
    ["null", null],
    ["a numeric code", { code: 5 }],
  ])("turns %s into a generic GIT_ERROR", (_label, value) => {
    const error = toRepositoryError(value);
    expect(error.code).toBe("GIT_ERROR");
    expect(error.message).toBe(ERROR_MESSAGES.GIT_ERROR);
  });

  it("passes an existing LocalRepositoryError through", () => {
    const original = new LocalRepositoryError("PERMISSION_DENIED");
    expect(toRepositoryError(original)).toBe(original);
  });
});

describe("selectLocalRepository", () => {
  it("returns the repository the desktop app reports", async () => {
    const inspect = vi.fn(async () => VALID_RESPONSE);
    const result = await selectLocalRepository(bridge({ inspectRepository: inspect }));
    expect(result).toEqual({ status: "selected", repository: TEST_REPOSITORY });
    expect(inspect).toHaveBeenCalledExactlyOnceWith(TEST_REPOSITORY.path);
  });

  it("uses one fixed command name", () => {
    expect(SELECT_REPOSITORY_COMMAND).toBe("select_repository");
  });

  it("fails clearly outside the desktop app and never picks or inspects anything", async () => {
    const pick = vi.fn(async () => "/x");
    const inspect = vi.fn(async () => VALID_RESPONSE);
    const error = await failureOf(selectLocalRepository(bridge({ isAvailable: () => false, pickDirectory: pick, inspectRepository: inspect })));
    expect(error.code).toBe("DESKTOP_REQUIRED");
    expect(error.message).toBe("Open Repository is available in the CodeFrog desktop app.");
    expect(pick).not.toHaveBeenCalled();
    expect(inspect).not.toHaveBeenCalled();
  });

  it("treats a cancelled picker as no selection, not an error, and inspects nothing", async () => {
    const inspect = vi.fn(async () => VALID_RESPONSE);
    const result = await selectLocalRepository(bridge({ pickDirectory: async () => null, inspectRepository: inspect }));
    expect(result).toEqual({ status: "cancelled" });
    expect(inspect).not.toHaveBeenCalled();
  });

  it("reports a picker failure", async () => {
    const error = await failureOf(
      selectLocalRepository(
        bridge({
          pickDirectory: async () => {
            throw new Error("dialog crashed at C:\\internal");
          },
        }),
      ),
    );
    expect(error.code).toBe("PICKER_FAILED");
    expect(error.message).not.toContain("internal");
  });

  it("rejects an empty path from the picker without calling the command", async () => {
    const inspect = vi.fn(async () => VALID_RESPONSE);
    const error = await failureOf(selectLocalRepository(bridge({ pickDirectory: async () => "  ", inspectRepository: inspect })));
    expect(error.code).toBe("INVALID_PATH");
    expect(inspect).not.toHaveBeenCalled();
  });

  it.each(["INVALID_PATH", "NOT_A_GIT_REPOSITORY", "PERMISSION_DENIED", "GIT_UNAVAILABLE"] satisfies RepositoryErrorCode[])(
    "maps the command error %s to a typed error",
    async (code) => {
      const error = await failureOf(
        selectLocalRepository(
          bridge({
            inspectRepository: async () => {
              throw { code, message: "raw OS text" };
            },
          }),
        ),
      );
      expect(error.code).toBe(code);
      expect(error.message).toBe(ERROR_MESSAGES[code]);
    },
  );

  it("maps an unrecognized command failure to a generic error", async () => {
    const error = await failureOf(
      selectLocalRepository(
        bridge({
          inspectRepository: async () => {
            throw "something unexpected with /home/user/private";
          },
        }),
      ),
    );
    expect(error.code).toBe("GIT_ERROR");
    expect(error.message).not.toContain("private");
  });

  it("rejects a malformed command result instead of storing partial data", async () => {
    const error = await failureOf(selectLocalRepository(bridge({ inspectRepository: async () => ({ name: "x" }) })));
    expect(error.code).toBe("INVALID_RESPONSE");
  });
});
