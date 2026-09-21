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

  it("only ever invokes the four intended commands, by constant, from the two service modules", () => {
    const invocations = UI_SOURCES.flatMap((file) => [...read(file).matchAll(/\binvoke\(([^)]*)\)/g)].map((match) => `${file}: ${match[1].split(",")[0].trim()}`));
    expect(invocations.sort()).toEqual([
      "lib/local-repository.ts: SELECT_REPOSITORY_COMMAND",
      "lib/repository-files.ts: CLEAR_SELECTION_COMMAND",
      "lib/repository-files.ts: LIST_TREE_COMMAND",
      "lib/repository-files.ts: READ_FILE_COMMAND",
    ]);
  });

  it("keeps components away from Tauri: they only use the service modules", () => {
    for (const file of sourceFiles("components")) {
      expect(read(file), file).not.toMatch(/@tauri-apps|\binvoke\b/);
    }
  });

  it("sends only a relative file path to the desktop app, never a root or an absolute path", () => {
    const service = read("lib/repository-files.ts");
    expect(service).toMatch(/invoke\(READ_FILE_COMMAND, \{ path \}\)/);
    expect(service).toMatch(/invoke\(LIST_TREE_COMMAND\)/);
    expect(service).toMatch(/isSafeRelativePath\(path\)/);
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
    const handler = lib.match(/generate_handler!\[([^\]]*)\]/)?.[1] ?? "";
    expect(handler.split(",").map((name) => name.trim()).filter(Boolean)).toEqual([
      "repository::select_repository",
      "repository_files::list_repository_tree",
      "repository_files::read_repository_file",
      "repository_files::clear_selected_repository",
    ]);
  });

  it("only starts the git program from Rust, never a shell or a caller-supplied program", () => {
    const rust = read("src-tauri/src/repository.rs");
    const programs = [...rust.matchAll(/Command::new\(([^)]*)\)/g)].map((match) => match[1]);
    expect(programs).toEqual(['"git"']);
    expect(rust).not.toMatch(/"(sh|bash|cmd|cmd\.exe|powershell|pwsh)"|\.arg\("-c"\)|env::vars|std::env::var/);
    // The path is the command's only parameter.
    expect(rust).toMatch(/pub async fn select_repository\(\s*path: String,/);
  });

  it("keeps repository file access read-only, process-free, and unlogged in Rust", () => {
    const rust = read("src-tauri/src/repository_files.rs").split("#[cfg(test)]")[0];
    expect(rust).not.toMatch(/std::process|Command::new|fs::(write|remove|create|rename|copy|set_permissions)|OpenOptions|File::create|std::env|env::var/);
    expect(rust).not.toMatch(/println!|eprintln!|dbg!|log::|tracing::/);
  });

  it("gives the file commands no root or absolute-path parameter: only a relative path", () => {
    const rust = read("src-tauri/src/repository_files.rs");
    expect(rust).toMatch(/pub async fn list_repository_tree\(\s*selection: State<'_, SelectedRepository>,\s*\)/);
    expect(rust).toMatch(/pub async fn read_repository_file\(\s*path: String,\s*selection: State<'_, SelectedRepository>,\s*\)/);
    expect(rust).toMatch(/pub fn clear_selected_repository\(selection: State<'_, SelectedRepository>\)/);
    // The root only ever comes from the selected repository.
    expect(rust.match(/selection\.get\(\)\?/g)).toHaveLength(2);
  });

  it("puts the same path boundary in front of reading files", () => {
    const rust = read("src-tauri/src/repository_files.rs");
    const reader = rust.slice(rust.indexOf("pub fn read_file("), rust.indexOf("fn too_large()"));
    expect(reader.indexOf("validate_relative_path(raw_path)")).toBeGreaterThan(-1);
    expect(reader.indexOf("validate_relative_path(raw_path)")).toBeLessThan(reader.indexOf("fs::canonicalize"));
    expect(reader.indexOf("fs::canonicalize")).toBeLessThan(reader.indexOf("strip_prefix(root)"));
    expect(reader.indexOf("strip_prefix(root)")).toBeLessThan(reader.indexOf("File::open"));
  });

  it("records the selected repository only after the folder passed validation", () => {
    const rust = read("src-tauri/src/repository.rs");
    expect(rust.indexOf("inspect_repository(&path)?")).toBeLessThan(rust.indexOf("selection.set("));
  });

  it("does not put credentials or keys in the desktop configuration", () => {
    for (const file of ["src-tauri/tauri.conf.json", "src-tauri/capabilities/default.json"]) {
      expect(read(file), file).not.toMatch(/sk-[A-Za-z0-9]{10}|api[_-]?key|token|secret/i);
    }
  });
});
