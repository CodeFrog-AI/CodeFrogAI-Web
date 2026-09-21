import { describe, expect, it, vi } from "vitest";

import { ApiError, type ApiDeps } from "@/lib/api";
import { appReducer, initialState } from "@/lib/app-state";
import { TEST_GITHUB_REPOSITORY, TEST_REPOSITORY } from "@/lib/fixtures";
import {
  connectGitHubRepository,
  filterRepositories,
  listGitHubRepositories,
  parseConnectedRepository,
  parseRepositoryList,
  parseScanSummary,
  scanRepository,
  toGitHubRepository,
  type GitHubRepositoryOption,
} from "@/lib/github-repositories";

const JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.c2lnbmF0dXJl";
const UUID = "3f2b8c1e-5d4a-4c6b-9e7f-1a2b3c4d5e6f";

const RAW_LIST = {
  repositories: [
    { github_repository_id: 1, owner: "shishir-21", name: "CodeFrogAI-Web", default_branch: "main", private: false, connected: true },
    { github_repository_id: 2, owner: "shishir-21", name: "MyOtherProject", default_branch: "develop", private: true, connected: false },
  ],
};
const RAW_CONNECTED = { id: UUID, github_repository_id: 2, owner: "shishir-21", name: "MyOtherProject", default_branch: "develop", private: true, connection_status: "connected" };

function deps(response: Response) {
  const fetchMock = vi.fn<ApiDeps["fetch"]>(async () => response);
  const clear = vi.fn();
  const value: ApiDeps = { fetch: fetchMock, baseUrl: () => "http://localhost:8000", getToken: () => JWT, clearSession: clear };
  return { value, fetchMock, clear };
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

async function failureOf(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    return error as ApiError;
  }
  throw new Error("expected failure");
}

describe("listing repositories", () => {
  it("loads the list with the CodeFrog JWT and parses it", async () => {
    const { value, fetchMock } = deps(json(RAW_LIST));

    const repositories = await listGitHubRepositories(value);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/v1/repositories/github");
    expect((init.headers as Record<string, string>).Authorization).toBe(`Bearer ${JWT}`);
    expect(repositories).toEqual([
      { githubRepositoryId: 1, owner: "shishir-21", name: "CodeFrogAI-Web", defaultBranch: "main", private: false, connected: true },
      { githubRepositoryId: 2, owner: "shishir-21", name: "MyOtherProject", defaultBranch: "develop", private: true, connected: false },
    ]);
  });

  it("accepts an empty list", () => {
    expect(parseRepositoryList({ repositories: [] })).toEqual([]);
  });

  it.each([
    ["null", null],
    ["a bare array", []],
    ["missing repositories", {}],
    ["a non-array", { repositories: {} }],
    ["a non-object item", { repositories: [5] }],
    ["a missing id", { repositories: [{ ...RAW_LIST.repositories[0], github_repository_id: undefined }] }],
    ["a string id", { repositories: [{ ...RAW_LIST.repositories[0], github_repository_id: "1" }] }],
    ["an id beyond the backend's range", { repositories: [{ ...RAW_LIST.repositories[0], github_repository_id: 3_000_000_000 }] }],
    ["a zero id", { repositories: [{ ...RAW_LIST.repositories[0], github_repository_id: 0 }] }],
    ["an empty name", { repositories: [{ ...RAW_LIST.repositories[0], name: "" }] }],
    ["a missing owner", { repositories: [{ ...RAW_LIST.repositories[0], owner: undefined }] }],
    ["a string private flag", { repositories: [{ ...RAW_LIST.repositories[0], private: "no" }] }],
    ["a missing connected flag", { repositories: [{ ...RAW_LIST.repositories[0], connected: undefined }] }],
  ])("rejects %s", (_label, value) => {
    expect(() => parseRepositoryList(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });

  it("clears the session on 401", async () => {
    const { value, clear } = deps(json({ detail: "x" }, 401));
    expect((await failureOf(listGitHubRepositories(value))).code).toBe("UNAUTHORIZED");
    expect(clear).toHaveBeenCalledOnce();
  });

  it("reports 403 as a reconnect situation, without clearing the session", async () => {
    const { value, clear } = deps(json({ detail: "x" }, 403));
    const error = await failureOf(listGitHubRepositories(value));
    expect(error.code).toBe("FORBIDDEN");
    expect(error.message).toContain("Reconnect GitHub");
    expect(clear).not.toHaveBeenCalled();
  });

  it("reports a backend failure with a fixed message", async () => {
    const { value } = deps(json({ detail: "Traceback ... psycopg.OperationalError" }, 502));
    const error = await failureOf(listGitHubRepositories(value));
    expect(error.code).toBe("SERVER_ERROR");
    expect(error.message).not.toMatch(/Traceback|psycopg/);
  });
});

describe("connecting a repository", () => {
  it("POSTs only the GitHub repository id and returns the CodeFrog repository", async () => {
    const { value, fetchMock } = deps(json(RAW_CONNECTED, 201));

    const connected = await connectGitHubRepository(2, value);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/v1/repositories/connect");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ github_repository_id: 2 });
    expect(connected).toEqual({ repositoryId: UUID, githubRepositoryId: 2, owner: "shishir-21", name: "MyOtherProject", defaultBranch: "develop", private: true });
  });

  it.each([0, -1, 1.5, Number.NaN, 3_000_000_000])("refuses the id %s without calling the backend", async (id) => {
    const { value, fetchMock } = deps(json(RAW_CONNECTED));
    expect((await failureOf(connectGitHubRepository(id, value))).code).toBe("BAD_REQUEST");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([
    ["a non-UUID id", { ...RAW_CONNECTED, id: "not-a-uuid" }],
    ["a missing id", { ...RAW_CONNECTED, id: undefined }],
    ["a missing owner", { ...RAW_CONNECTED, owner: undefined }],
    ["a string private flag", { ...RAW_CONNECTED, private: "yes" }],
    ["null", null],
  ])("rejects a response with %s", (_label, value) => {
    expect(() => parseConnectedRepository(value)).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });

  it.each([
    [403, "FORBIDDEN"],
    [404, "NOT_FOUND"],
    [409, "CONFLICT"],
    [500, "SERVER_ERROR"],
  ])("maps HTTP %d to %s", async (status, code) => {
    const { value } = deps(json({ detail: "x" }, status));
    expect((await failureOf(connectGitHubRepository(2, value))).code).toBe(code);
  });

  it("clears the session on 401", async () => {
    const { value, clear } = deps(json({}, 401));
    expect((await failureOf(connectGitHubRepository(2, value))).code).toBe("UNAUTHORIZED");
    expect(clear).toHaveBeenCalledOnce();
  });
});

describe("selecting a connected repository", () => {
  it("becomes GitHub-sourced application state with the CodeFrog id", () => {
    const repository = toGitHubRepository(parseConnectedRepository(RAW_CONNECTED));
    expect(repository).toEqual({
      source: "github",
      repositoryId: UUID,
      githubRepositoryId: 2,
      owner: "shishir-21",
      name: "MyOtherProject",
      defaultBranch: "develop",
      private: true,
    });
    const state = appReducer(initialState, { type: "select-repository", repository });
    expect(state.repository).toEqual(repository);
  });

  it("keeps GitHub and local repositories apart", () => {
    const github = appReducer(initialState, { type: "select-repository", repository: TEST_GITHUB_REPOSITORY });
    expect(github.repository?.source).toBe("github");
    expect(github.repository && "path" in github.repository).toBe(false);

    const local = appReducer(github, { type: "select-repository", repository: TEST_REPOSITORY });
    expect(local.repository?.source).toBe("local");
    expect(local.repository && "repositoryId" in local.repository).toBe(false);
    expect(local.repository && "githubRepositoryId" in local.repository).toBe(false);
  });

  it("can be cleared to pick another repository", () => {
    const selected = appReducer(initialState, { type: "select-repository", repository: TEST_GITHUB_REPOSITORY });
    expect(appReducer(selected, { type: "clear-repository" }).repository).toBeNull();
  });
});

describe("filterRepositories", () => {
  const repositories: GitHubRepositoryOption[] = parseRepositoryList(RAW_LIST);

  it("returns everything for an empty query", () => {
    expect(filterRepositories(repositories, "")).toHaveLength(2);
    expect(filterRepositories(repositories, "   ")).toHaveLength(2);
  });

  it("matches owner/name case-insensitively", () => {
    expect(filterRepositories(repositories, "codefrog").map((r) => r.name)).toEqual(["CodeFrogAI-Web"]);
    expect(filterRepositories(repositories, "SHISHIR-21/my").map((r) => r.name)).toEqual(["MyOtherProject"]);
    expect(filterRepositories(repositories, "shishir")).toHaveLength(2);
  });

  it("returns nothing for no match, and does not modify the input", () => {
    expect(filterRepositories(repositories, "zzz")).toEqual([]);
    expect(repositories).toHaveLength(2);
  });
});

describe("scanning a repository", () => {
  const RAW_SCAN = { repository_id: UUID, status: "completed", files_discovered: 10, files_indexed: 8, files_skipped: 2, files_removed: 0, chunks_created: 30, embeddings: {} };

  it("POSTs to the scan endpoint of the CodeFrog repository and parses the counts", async () => {
    const { value, fetchMock } = deps(json(RAW_SCAN));

    const summary = await scanRepository(UUID, value);

    expect(fetchMock.mock.calls[0][0]).toBe(`http://localhost:8000/api/v1/repositories/${UUID}/scan`);
    expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    expect(summary).toEqual({ filesDiscovered: 10, filesIndexed: 8, filesSkipped: 2, filesRemoved: 0, chunksCreated: 30 });
  });

  it.each(["", "abc", "../../etc", UUID + "/../x", "1; DROP TABLE"])("refuses the repository id %j", async (id) => {
    const { value, fetchMock } = deps(json(RAW_SCAN));
    expect((await failureOf(scanRepository(id, value))).code).toBe("BAD_REQUEST");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects a malformed scan response", () => {
    expect(() => parseScanSummary({ files_discovered: "many" })).toThrowError(expect.objectContaining({ code: "INVALID_RESPONSE" }));
  });
});
