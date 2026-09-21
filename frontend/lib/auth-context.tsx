"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { ApiError, getCurrentUser, type CurrentUser } from "@/lib/api";
import { clearSession, getSessionToken } from "@/lib/session";

export type AuthState =
  | { status: "loading" }
  /** `expired` is true when a session existed but the backend rejected it (401). */
  | { status: "signed-out"; expired: boolean }
  | { status: "signed-in"; user: CurrentUser }
  | { status: "error"; error: ApiError };

interface AuthContextValue {
  state: AuthState;
  /** Ask the backend who the current session belongs to. */
  refresh: () => Promise<void>;
  setSignedIn: (user: CurrentUser) => void;
  /** The backend rejected the session: forget it. */
  expire: () => void;
  signOut: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

/** The OAuth callback page finishes sign-in itself; the provider must not race it. */
const CALLBACK_PATH = "/auth/callback";

/** Who the current session belongs to, according to the backend. Never throws. */
async function loadAuthState(): Promise<AuthState> {
  if (getSessionToken() === null) return { status: "signed-out", expired: false };
  try {
    return { status: "signed-in", user: await getCurrentUser() };
  } catch (error) {
    const apiError = error instanceof ApiError ? error : new ApiError("SERVER_ERROR");
    return apiError.code === "UNAUTHORIZED" ? { status: "signed-out", expired: true } : { status: "error", error: apiError };
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({ status: "loading" });

  const refresh = useCallback(async () => {
    setState({ status: "loading" });
    setState(await loadAuthState());
  }, []);

  useEffect(() => {
    if (window.location.pathname.startsWith(CALLBACK_PATH)) return;
    let cancelled = false;
    void loadAuthState().then((next) => {
      if (!cancelled) setState(next);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const setSignedIn = useCallback((user: CurrentUser) => setState({ status: "signed-in", user }), []);
  const expire = useCallback(() => {
    clearSession();
    setState({ status: "signed-out", expired: true });
  }, []);
  const signOut = useCallback(() => {
    clearSession();
    setState({ status: "signed-out", expired: false });
  }, []);

  // The actions are stable, so effects that depend on them do not re-run when the state changes.
  const value = useMemo<AuthContextValue>(() => ({ state, refresh, setSignedIn, expire, signOut }), [state, refresh, setSignedIn, expire, signOut]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (context === null) {
    throw new Error("useAuth must be used inside <AuthProvider>");
  }
  return context;
}
