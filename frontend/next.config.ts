import type { NextConfig } from "next";

// The desktop app (Tauri) loads a static export. The regular web build is unchanged.
const isDesktopBuild = process.env.CODEFROG_DESKTOP === "1";

const nextConfig: NextConfig = {
  ...(isDesktopBuild ? { output: "export", images: { unoptimized: true } } : {}),
};

export default nextConfig;
