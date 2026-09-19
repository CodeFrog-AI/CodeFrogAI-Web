import { describe, expect, it } from "vitest";

import { describeLanguages, NOT_ANALYZED, SAMPLE_REPOSITORY } from "@/lib/repository";

describe("describeLanguages", () => {
  it("shows a placeholder when nothing is detected", () => {
    expect(describeLanguages([])).toBe(NOT_ANALYZED);
  });

  it("joins detected languages", () => {
    expect(describeLanguages(["Python", "TypeScript"])).toBe("Python, TypeScript");
  });
});

describe("SAMPLE_REPOSITORY", () => {
  it("is clearly placeholder data with nothing analyzed", () => {
    expect(SAMPLE_REPOSITORY.status).toBe(NOT_ANALYZED);
    expect(SAMPLE_REPOSITORY.projectType).toBe(NOT_ANALYZED);
    expect(SAMPLE_REPOSITORY.languages).toEqual([]);
  });

  it("points at no real absolute path or credential", () => {
    expect(SAMPLE_REPOSITORY.path.startsWith("~/")).toBe(true);
    expect(JSON.stringify(SAMPLE_REPOSITORY)).not.toMatch(/key|token|secret/i);
  });
});
