"use client";

import type { ReactNode } from "react";

import { AppProvider } from "@/lib/app-context";
import { AuthProvider } from "@/lib/auth-context";

/** App-wide state, mounted once in the root layout so it survives the OAuth callback navigation. */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <AuthProvider>
      <AppProvider>{children}</AppProvider>
    </AuthProvider>
  );
}
