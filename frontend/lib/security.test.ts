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
  it("never uses localStorage, IndexedDB, or cookies, so nothing (paths, contents, keys) is persisted there", () => {
    for (const file of UI_SOURCES) {
      expect(read(file), file).not.toMatch(/\b(localStorage|indexedDB|openDatabase)\b|document\.cookie/);
    }
  });

  it("uses sessionStorage only in lib/session.ts, and only for the CodeFrog JWT", () => {
    for (const file of UI_SOURCES) {
      const usesSessionStorage = /\bsessionStorage\b/.test(read(file));
      expect(usesSessionStorage, file).toBe(file === "lib/session.ts");
    }
    const session = read("lib/session.ts");
    expect(session.match(/setItem\(/g)).toHaveLength(2); // the interface declaration and the single call
    expect(session).toMatch(/storage\.setItem\(SESSION_KEY, token\)/);
    expect(session).toMatch(/SESSION_KEY = "codefrog\.access_token"/);
  });

  it("holds no GitHub access token, client secret, or provider credential in the frontend", () => {
    for (const file of UI_SOURCES) {
      expect(read(file), file).not.toMatch(/github_access_token|GITHUB_CLIENT_SECRET|client_secret|\bgh[opsu]_[A-Za-z0-9]{10}|GITHUB_CLIENT_ID/i);
    }
  });

  it("reads exactly one public environment variable: the backend URL", () => {
    const variables = UI_SOURCES.flatMap((file) => [...read(file).matchAll(/process\.env\.([A-Z0-9_]+)/g)].map((match) => match[1]));
    expect([...new Set(variables)]).toEqual(["NEXT_PUBLIC_API_URL"]);
  });

  it("talks to the network only through the API client (plus the existing backend health check)", () => {
    const callers = UI_SOURCES.filter((file) => /\bfetch\(/.test(read(file))).sort();
    expect(callers).toEqual(["app/backend-check/page.tsx", "lib/api.ts"]);
    for (const file of sourceFiles("components")) {
      expect(read(file), file).not.toMatch(/https?:\/\/(localhost|127\.0\.0\.1)|XMLHttpRequest|WebSocket/);
    }
  });

  it("starts GitHub OAuth with a full-page navigation, not fetch, a popup, or an iframe", () => {
    const auth = read("lib/github-auth.ts");
    expect(auth).toMatch(/window\.location\.assign\(url\)/);
    expect(auth).not.toMatch(/\bfetch\(|window\.open|<iframe/);
    for (const file of sourceFiles("components")) {
      expect(read(file), file).not.toMatch(/window\.open|<iframe|githubLoginUrl/);
    }
    // Connect GitHub buttons all call the one helper.
    for (const file of ["components/github/GitHubConnectionCard.tsx", "components/github/GitHubRepositoryPicker.tsx", "components/pages/RepositoriesPage.tsx"]) {
      expect(read(file), file).toMatch(/connectGitHub\(\)/);
    }
  });

  it("never logs, renders, or leaves the JWT in the URL", () => {
    for (const file of ["lib/api.ts", "lib/session.ts", "lib/github-auth.ts", "lib/github-repositories.ts", "lib/auth-context.tsx", "app/auth/callback/page.tsx"]) {
      expect(read(file), file).not.toMatch(/console\.|debugger/);
    }
    const page = read("app/auth/callback/page.tsx");
    expect(page).not.toMatch(/access_token|searchParams|location\.search|\{token\}/);
    expect(page).toMatch(/replaceState/);
    const auth = read("lib/github-auth.ts");
    // The URL is cleaned before the token is stored or sent anywhere.
    expect(auth.indexOf("deps.replaceUrl()")).toBeGreaterThan(-1);
    expect(auth.indexOf("deps.replaceUrl()")).toBeLessThan(auth.indexOf("setSessionToken(outcome.token"));
    // The token only ever goes out as a Bearer header.
    const api = read("lib/api.ts");
    expect(api.match(/\$\{token\}/g)).toHaveLength(1);
    expect(api).toMatch(/Authorization = `Bearer \$\{token\}`/);
    expect(UI_SOURCES.filter((file) => /[?&]access_token=/.test(read(file)))).toEqual([]);
  });

  it("sends AI provider keys only to /api/v1/settings/ai, never in a URL, and never stores them", () => {
    const settings = read("lib/ai-settings.ts");
    const paths = [...settings.matchAll(/"(\/api\/v1\/[^"]*)"/g)].map((match) => match[1]);
    expect(paths).toEqual(["/api/v1/settings/ai"]);
    expect(settings).toMatch(/LLM_KEY_PATH = `\$\{AI_SETTINGS_PATH\}\/llm-key`/);
    expect(settings).toMatch(/EMBEDDING_KEY_PATH = `\$\{AI_SETTINGS_PATH\}\/embedding-key`/);
    expect(settings).not.toMatch(/localStorage|sessionStorage|console\./);
    // Only the PUT carries a body, and no request path is built from a key.
    expect(settings.match(/body:/g)).toHaveLength(1);
    expect(settings).not.toMatch(/`[^`]*\$\{[^}]*[Kk]ey[^}]*\}[^`]*`/);
    const page = read("components/pages/SettingsPage.tsx");
    expect(page).not.toMatch(/localStorage|sessionStorage|indexedDB|document\.cookie|console\.|\bfetch\(/);
    expect(page.match(/type="password"/g)).toHaveLength(2);
    expect(page.match(/type="password"\s+autoComplete="off"/g)).toHaveLength(2);
    // No user-configurable provider address.
    expect(`${page}${settings}`).not.toMatch(/base_?url|baseUrl/i);
    for (const file of UI_SOURCES.filter((f) => f !== "lib/ai-settings.ts" && f !== "lib/api.ts")) {
      expect(read(file), file).not.toMatch(/\/api\/v1\/settings\/ai/);
    }
  });

  it("only accepts fixed backend API paths", () => {
    expect(read("lib/api.ts")).toMatch(/API_PATH = \/\^\\\/api\\\/v1\\\//);
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
