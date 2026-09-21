import { describe, expect, it, vi } from "vitest";

import {
  clearRepositorySelection,
  FILES_ERROR_MESSAGES,
  formatFileSize,
  isSafeRelativePath,
  listRepositoryTree,
  parseRepositoryFile,
  parseRepositoryTree,
  readRepositoryFile,
  RepositoryFilesError,
  toFilesError,
  type FilesBridge,
  type RepositoryFilesErrorCode,
} from "@/lib/repository-files";

const NUL = String.fromCharCode(0);

/** The shape the desktop app returns for a small repository. */
const RAW_TREE = {
  entries: [
    {
      name: "src",
      path: "src",
      type: "directory",
      children: [
        { name: "app", path: "src/app", type: "directory", children: [{ name: "page.tsx", path: "src/app/page.tsx", type: "file", size: 120 }] },
        { name: "index.ts", path: "src/index.ts", type: "file", size: 8 },
      ],
    },
    { name: "README.md", path: "README.md", type: "file", size: 2048 },
  ],
  truncated: false,
  entryCount: 5,
};
const RAW_FILE = { path: "src/index.ts", name: "index.ts", size: 8, contents: "export {};" };

function bridge(overrides: Partial<FilesBridge> = {}): FilesBridge {
  return {
    isAvailable: () => true,
    listTree: async () => RAW_TREE,
    readFile: async () => RAW_FILE,
    clearSelection: async () => undefined,
    ...overrides,
  };
}

async function failureOf(promise: Promise<unknown>): Promise<RepositoryFilesError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(RepositoryFilesError);
    return error as RepositoryFilesError;
  }
  throw new Error("expected the call to fail");
}

describe("parseRepositoryTree", () => {
  it("parses a valid tree into typed nodes", () => {
    const tree = parseRepositoryTree(RAW_TREE);
    expect(tree.truncated).toBe(false);
    expect(tree.entryCount).toBe(5);
    expect(tree.entries.map((entry) => `${entry.type}:${entry.path}`)).toEqual(["directory:src", "file:README.md"]);
    const src = tree.entries[0];
    expect(src.type === "directory" && src.children.map((child) => child.path)).toEqual(["src/app", "src/index.ts"]);
  });

  it("keeps the order the desktop app sent (it is already sorted)", () => {
    const parsed = parseRepositoryTree({ ...RAW_TREE, entries: [...RAW_TREE.entries].reverse() });
    expect(parsed.entries.map((entry) => entry.name)).toEqual(["README.md", "src"]);
  });

  it("accepts an empty repository and a truncated listing", () => {
    expect(parseRepositoryTree({ entries: [], truncated: true, entryCount: 0 })).toEqual({ entries: [], truncated: true, entryCount: 0 });
  });

  it("accepts an empty directory", () => {
    const tree = parseRepositoryTree({ entries: [{ name: "d", path: "d", type: "directory", children: [] }], truncated: false, entryCount: 1 });
    expect(tree.entries[0]).toEqual({ type: "directory", name: "d", path: "d", children: [] });
  });

  it.each([
    ["null", null],
    ["an array", []],
    ["a string", "tree"],
    ["missing entries", { truncated: false, entryCount: 0 }],
    ["entries that are not a list", { entries: {}, truncated: false, entryCount: 0 }],
    ["a missing truncated flag", { entries: [], entryCount: 0 }],
    ["a string truncated flag", { entries: [], truncated: "no", entryCount: 0 }],
    ["a negative entry count", { entries: [], truncated: false, entryCount: -1 }],
    ["a fractional entry count", { entries: [], truncated: false, entryCount: 1.5 }],
  ])("rejects %s", (_label, value) => {
    expect(() => parseRepositoryTree(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });

  const node = (overrides: Record<string, unknown>) => ({
    entries: [{ name: "a.ts", path: "a.ts", type: "file", size: 1, ...overrides }],
    truncated: false,
    entryCount: 1,
  });

  it.each([
    ["an unknown node type", node({ type: "symlink" })],
    ["a missing type", node({ type: undefined })],
    ["a missing name", node({ name: undefined })],
    ["an empty name", node({ name: "" })],
    ["a name containing a slash", node({ name: "a/b.ts", path: "a/b.ts" })],
    ["a name with a NUL", node({ name: "a" + NUL, path: "a" + NUL })],
    ["a path that does not match the name", node({ path: "other.ts" })],
    ["an absolute path", node({ path: "/etc/passwd", name: "passwd" })],
    ["a path with traversal", node({ path: "../a.ts" })],
    ["a Windows drive path", node({ path: "C:/a.ts", name: "a.ts" })],
    ["a backslash path", node({ path: "a\\b.ts", name: "b.ts" })],
    ["a file without a size", node({ size: undefined })],
    ["a string size", node({ size: "1" })],
    ["a negative size", node({ size: -5 })],
    ["a file with children", node({ children: [] })],
    ["a directory without children", node({ type: "directory", size: undefined, children: undefined })],
    ["a directory with a size", node({ type: "directory", children: [] })],
    ["a directory whose children are not a list", node({ type: "directory", size: undefined, children: "x" })],
  ])("rejects %s", (_label, value) => {
    expect(() => parseRepositoryTree(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });

  it("rejects a child whose path is not under its parent", () => {
    const value = {
      entries: [{ name: "src", path: "src", type: "directory", children: [{ name: "x.ts", path: "other/x.ts", type: "file", size: 1 }] }],
      truncated: false,
      entryCount: 2,
    };
    expect(() => parseRepositoryTree(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });

  it("rejects an absurdly deep or large tree instead of hanging", () => {
    let deep: Record<string, unknown> = { name: "f", path: "d0/f", type: "file", size: 1 };
    for (let level = 80; level >= 0; level -= 1) {
      const path = Array.from({ length: level + 1 }, (_, index) => `d${index}`).join("/");
      deep = { name: `d${level}`, path, type: "directory", children: [deep] };
    }
    expect(() => parseRepositoryTree({ entries: [deep], truncated: false, entryCount: 1 })).toThrow();
  });
});

describe("parseRepositoryFile", () => {
  it("parses a valid file and preserves its contents exactly", () => {
    const contents = "line 1\n\tindented\r\n  trailing  \n";
    expect(parseRepositoryFile({ ...RAW_FILE, contents })).toEqual({ ...RAW_FILE, contents });
  });

  it("accepts an empty file", () => {
    expect(parseRepositoryFile({ ...RAW_FILE, size: 0, contents: "" }).contents).toBe("");
  });

  it.each([
    ["null", null],
    ["a missing path", { ...RAW_FILE, path: undefined }],
    ["an unsafe path", { ...RAW_FILE, path: "../index.ts", name: "index.ts" }],
    ["a name that is not the path's last segment", { ...RAW_FILE, name: "other.ts" }],
    ["a name containing a slash", { ...RAW_FILE, name: "src/index.ts" }],
    ["a string size", { ...RAW_FILE, size: "8" }],
    ["a negative size", { ...RAW_FILE, size: -1 }],
    ["non-string contents", { ...RAW_FILE, contents: 42 }],
    ["missing contents", { ...RAW_FILE, contents: undefined }],
  ])("rejects %s", (_label, value) => {
    expect(() => parseRepositoryFile(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });
});

describe("isSafeRelativePath", () => {
  it.each(["a", "src/app/page.tsx", "dir/.hidden", "a b/c.txt"])("accepts %s", (path) => {
    expect(isSafeRelativePath(path)).toBe(true);
  });

  it.each([
    "",
    "/etc/passwd",
    "../../.env",
    "..",
    "a/../b",
    "./a",
    "a//b",
    "a/",
    "C:/x",
    "a\\b",
    "a" + NUL + "b",
    "bell" + String.fromCharCode(7),
    "x".repeat(2000),
  ])("rejects %j", (path) => {
    expect(isSafeRelativePath(path)).toBe(false);
  });

  it("rejects non-strings", () => {
    expect(isSafeRelativePath(null)).toBe(false);
    expect(isSafeRelativePath(5)).toBe(false);
  });
});

describe("listRepositoryTree", () => {
  it("returns the parsed tree", async () => {
    const tree = await listRepositoryTree(bridge());
    expect(tree.entries).toHaveLength(2);
  });

  it("fails clearly outside the desktop app without calling it", async () => {
    const list = vi.fn(async () => RAW_TREE);
    const error = await failureOf(listRepositoryTree(bridge({ isAvailable: () => false, listTree: list })));
    expect(error.code).toBe("DESKTOP_REQUIRED");
    expect(error.message).toBe("Repository file browsing is available in the CodeFrog desktop app.");
    expect(list).not.toHaveBeenCalled();
  });

  it("rejects a malformed response", async () => {
    const error = await failureOf(listRepositoryTree(bridge({ listTree: async () => ({ entries: "nope" }) })));
    expect(error.code).toBe("INVALID_RESPONSE");
  });

  it("maps a structured command error to a typed error with a fixed message", async () => {
    const error = await failureOf(
      listRepositoryTree(
        bridge({
          listTree: async () => {
            throw { code: "NO_REPOSITORY_SELECTED", message: "raw text C:\\Users\\me" };
          },
        }),
      ),
    );
    expect(error.code).toBe("NO_REPOSITORY_SELECTED");
    expect(error.message).toBe(FILES_ERROR_MESSAGES.NO_REPOSITORY_SELECTED);
    expect(error.message).not.toContain("Users");
  });

  it("maps an unrecognized failure to a generic error", async () => {
    const error = await failureOf(
      listRepositoryTree(
        bridge({
          listTree: async () => {
            throw new Error("boom at /home/user/secret");
          },
        }),
      ),
    );
    expect(error.code).toBe("IO_ERROR");
    expect(error.message).not.toContain("secret");
  });
});

describe("readRepositoryFile", () => {
  it("passes only the relative path to the desktop app and returns the parsed file", async () => {
    const read = vi.fn(async () => RAW_FILE);
    const file = await readRepositoryFile("src/index.ts", bridge({ readFile: read }));
    expect(file).toEqual(RAW_FILE);
    expect(read).toHaveBeenCalledExactlyOnceWith("src/index.ts");
  });

  it("fails clearly outside the desktop app without calling it", async () => {
    const read = vi.fn(async () => RAW_FILE);
    const error = await failureOf(readRepositoryFile("a.ts", bridge({ isAvailable: () => false, readFile: read })));
    expect(error.code).toBe("DESKTOP_REQUIRED");
    expect(read).not.toHaveBeenCalled();
  });

  it.each(["../../.env", "/etc/passwd", "C:/Windows/win.ini", "a\\..\\b", "", "a/../../b"])(
    "refuses %j before calling the desktop app",
    async (path) => {
      const read = vi.fn(async () => RAW_FILE);
      const error = await failureOf(readRepositoryFile(path, bridge({ readFile: read })));
      expect(error.code).toBe("INVALID_PATH");
      expect(read).not.toHaveBeenCalled();
    },
  );

  it.each(["FILE_TOO_LARGE", "NOT_UTF8_TEXT", "NOT_FOUND", "NOT_A_FILE", "PATH_OUTSIDE_REPOSITORY", "PERMISSION_DENIED"] satisfies RepositoryFilesErrorCode[])(
    "maps the command error %s",
    async (code) => {
      const error = await failureOf(
        readRepositoryFile(
          "a.ts",
          bridge({
            readFile: async () => {
              throw { code, message: "raw" };
            },
          }),
        ),
      );
      expect(error.code).toBe(code);
      expect(error.message).toBe(FILES_ERROR_MESSAGES[code]);
    },
  );

  it("rejects a malformed file response", async () => {
    const error = await failureOf(readRepositoryFile("src/index.ts", bridge({ readFile: async () => ({ path: "src/index.ts" }) })));
    expect(error.code).toBe("INVALID_RESPONSE");
  });

  it("rejects an answer for a different file than the one requested", async () => {
    const error = await failureOf(readRepositoryFile("src/other.ts", bridge({ readFile: async () => RAW_FILE })));
    expect(error.code).toBe("INVALID_RESPONSE");
  });
});

describe("toFilesError", () => {
  it("passes an existing error through", () => {
    const original = new RepositoryFilesError("NOT_FOUND");
    expect(toFilesError(original)).toBe(original);
  });

  it.each([["an unknown code", { code: "WHAT" }], ["a string", "oops"], ["null", null], ["a numeric code", { code: 4 }]])(
    "turns %s into IO_ERROR",
    (_label, value) => {
      expect(toFilesError(value).code).toBe("IO_ERROR");
    },
  );
});

describe("clearRepositorySelection", () => {
  it("asks the desktop app to forget the repository", async () => {
    const clear = vi.fn(async () => undefined);
    await clearRepositorySelection(bridge({ clearSelection: clear }));
    expect(clear).toHaveBeenCalledOnce();
  });

  it("does nothing outside the desktop app", async () => {
    const clear = vi.fn(async () => undefined);
    await clearRepositorySelection(bridge({ isAvailable: () => false, clearSelection: clear }));
    expect(clear).not.toHaveBeenCalled();
  });

  it("never throws", async () => {
    await expect(
      clearRepositorySelection(
        bridge({
          clearSelection: async () => {
            throw new Error("nope");
          },
        }),
      ),
    ).resolves.toBeUndefined();
  });
});

describe("formatFileSize", () => {
  it.each([
    [0, "0 B"],
    [1023, "1023 B"],
    [1024, "1.0 KB"],
    [1536, "1.5 KB"],
    [1024 * 1024, "1.0 MB"],
    [2.5 * 1024 * 1024, "2.5 MB"],
  ])("%d bytes", (bytes, expected) => {
    expect(formatFileSize(bytes)).toBe(expected);
  });
});
