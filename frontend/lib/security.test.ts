import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const ROOT = fileURLToPath(new URL("../", import.meta.url));

function read(relative: string): string {
  return readFileSync(join(ROOT, relative), "utf8");
}

function sourceFiles(directory: string): string[] {
  return readdirSync(join(ROOT, directory)).flatMap((entry) => {
    const relative = `${directory}/${entry}`;
    if (statSync(join(ROOT, relative)).isDirectory()) return sourceFiles(relative);
    return /\.(ts|tsx)$/.test(entry) && !/\.test\.ts$/.test(entry) ? [relative] : [];
  });
}

const UI_SOURCES = [...sourceFiles("app"), ...sourceFiles("components"), ...sourceFiles("lib")];

describe("frontend security guarantees", () => {
  it("never uses browser storage, so nothing (keys, paths, contents) is persisted", () => {
    for (const file of UI_SOURCES) {
      expect(read(file), file).not.toMatch(/\b(localStorage|sessionStorage|indexedDB)\b|document\.cookie/);
    }
  });

  it("only ever invokes the single select_repository command", () => {
    const invocations = UI_SOURCES.flatMap((file) => [...read(file).matchAll(/\binvoke\(([^)]*)\)/g)].map((match) => [file, match[1]]));
    expect(invocations).toHaveLength(1);
    expect(invocations[0][0]).toBe("lib/local-repository.ts");
    expect(invocations[0][1]).toMatch(/^SELECT_REPOSITORY_COMMAND\b/);
  });

  it("does not depend on filesystem, shell, process, or HTTP Tauri plugins", () => {
    const packageJson = JSON.parse(read("package.json")) as { dependencies: Record<string, string>; devDependencies: Record<string, string> };
    const tauriPackages = Object.keys({ ...packageJson.dependencies, ...packageJson.devDependencies }).filter((name) => name.startsWith("@tauri-apps/"));
    expect(tauriPackages.sort()).toEqual(["@tauri-apps/api", "@tauri-apps/cli", "@tauri-apps/plugin-dialog"]);
  });

  it("grants the main window only core access and the folder picker", () => {
    const capability = JSON.parse(read("src-tauri/capabilities/default.json")) as { permissions: string[]; windows: string[] };
    expect(capability.permissions).toEqual(["core:default", "dialog:allow-open"]);
    expect(capability.windows).toEqual(["main"]);
  });

  it("registers no native plugin other than the dialog plugin", () => {
    const cargo = read("src-tauri/Cargo.toml");
    const dependencies = cargo.split("[dependencies]")[1].split("[dev-dependencies]")[0];
    const names = [...dependencies.matchAll(/^([a-z0-9_-]+)\s*=/gm)].map((match) => match[1]);
    expect(names.sort()).toEqual(["serde", "tauri", "tauri-plugin-dialog"]);
    const lib = read("src-tauri/src/lib.rs");
    expect([...lib.matchAll(/\.plugin\((.+)\)\s*$/gm)].map((match) => match[1])).toEqual(["tauri_plugin_dialog::init()"]);
    expect(lib).toMatch(/generate_handler!\[repository::select_repository\]/);
  });

  it("only starts the git program from Rust, never a shell or a caller-supplied program", () => {
    const rust = read("src-tauri/src/repository.rs");
    const programs = [...rust.matchAll(/Command::new\(([^)]*)\)/g)].map((match) => match[1]);
    expect(programs).toEqual(['"git"']);
    expect(rust).not.toMatch(/"(sh|bash|cmd|cmd\.exe|powershell|pwsh)"|\.arg\("-c"\)|env::vars|std::env::var/);
    // The path is the command's only parameter.
    expect(rust).toMatch(/pub async fn select_repository\(path: String\)/);
  });

  it("does not put credentials or keys in the desktop configuration", () => {
    for (const file of ["src-tauri/tauri.conf.json", "src-tauri/capabilities/default.json"]) {
      expect(read(file), file).not.toMatch(/sk-[A-Za-z0-9]{10}|api[_-]?key|token|secret/i);
    }
  });
});
