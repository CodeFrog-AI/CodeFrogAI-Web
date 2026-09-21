"use client";

import { createContext, useContext, useMemo, useReducer, type ReactNode } from "react";

import { appReducer, initialState, type AppAction, type AppState } from "@/lib/app-state";

interface AppContextValue {
  state: AppState;
  dispatch: (action: AppAction) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppContextValue {
  const context = useContext(AppContext);
  if (context === null) {
    throw new Error("useApp must be used inside <AppProvider>");
  }
  return context;
}
