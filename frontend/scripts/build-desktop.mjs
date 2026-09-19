// Builds the static export the Tauri app loads (frontend/out). Cross-platform, no shell.
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const nextBin = createRequire(import.meta.url).resolve("next/dist/bin/next");
const result = spawnSync(process.execPath, [nextBin, "build"], {
  stdio: "inherit",
  env: { ...process.env, CODEFROG_DESKTOP: "1" },
});
process.exit(result.status ?? 1);
