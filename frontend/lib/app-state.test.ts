import { describe, expect, it } from "vitest";

import { appReducer, initialState, isPage, PAGES, pageDefinition, type AppState } from "@/lib/app-state";
import { SAMPLE_REPOSITORY } from "@/lib/repository";

describe("appReducer", () => {
  it("starts on the repositories page with nothing selected", () => {
    expect(initialState).toEqual({ page: "repositories", repository: null });
  });

  it("navigates between pages", () => {
    expect(appReducer(initialState, { type: "navigate", page: "agent" }).page).toBe("agent");
  });

  it("returns the same state object when navigating to the current page", () => {
    expect(appReducer(initialState, { type: "navigate", page: "repositories" })).toBe(initialState);
  });

  it("ignores navigation to an unknown page", () => {
    const bogus = { type: "navigate", page: "terminal" } as unknown as Parameters<typeof appReducer>[1];
    expect(appReducer(initialState, bogus)).toBe(initialState);
  });

  it("selects and clears a repository without changing the page", () => {
    const onAgent: AppState = { ...initialState, page: "agent" };
    const selected = appReducer(onAgent, { type: "select-repository", repository: SAMPLE_REPOSITORY });
    expect(selected).toEqual({ page: "agent", repository: SAMPLE_REPOSITORY });
    expect(appReducer(selected, { type: "clear-repository" })).toEqual(onAgent);
  });

  it("does not create a new state when clearing an empty selection", () => {
    expect(appReducer(initialState, { type: "clear-repository" })).toBe(initialState);
  });

  it("does not mutate the previous state", () => {
    const before = structuredClone(initialState);
    appReducer(initialState, { type: "select-repository", repository: SAMPLE_REPOSITORY });
    expect(initialState).toEqual(before);
  });
});

describe("pages", () => {
  it("lists the four navigation targets in order", () => {
    expect(PAGES.map((page) => page.label)).toEqual(["Repositories", "Agent", "Pull Requests", "Settings"]);
  });

  it("has unique ids", () => {
    expect(new Set(PAGES.map((page) => page.id)).size).toBe(PAGES.length);
  });

  it("recognizes only real pages", () => {
    expect(isPage("settings")).toBe(true);
    expect(isPage("terminal")).toBe(false);
    expect(isPage(undefined)).toBe(false);
  });

  it("looks up a page definition", () => {
    expect(pageDefinition("pull-requests").label).toBe("Pull Requests");
  });
});
