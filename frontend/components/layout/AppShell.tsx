"use client";

import { AgentPage } from "@/components/pages/AgentPage";
import { PullRequestsPage } from "@/components/pages/PullRequestsPage";
import { RepositoriesPage } from "@/components/pages/RepositoriesPage";
import { SettingsPage } from "@/components/pages/SettingsPage";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { useApp } from "@/lib/app-context";
import { pageDefinition, type Page } from "@/lib/app-state";

const PAGE_COMPONENTS: Record<Page, () => React.JSX.Element> = {
  repositories: RepositoriesPage,
  agent: AgentPage,
  "pull-requests": PullRequestsPage,
  settings: SettingsPage,
};

/** Sidebar on the left; header above the main content on the right. */
export function AppShell() {
  const { state, dispatch } = useApp();
  const CurrentPage = PAGE_COMPONENTS[state.page];

  return (
    <div className="flex h-full min-h-0">
      <a
        href="#main"
        className="sr-only rounded-md bg-accent px-3 py-2 text-sm text-accent-foreground focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-10"
      >
        Skip to content
      </a>
      <Sidebar current={state.page} onNavigate={(page) => dispatch({ type: "navigate", page })} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header title={pageDefinition(state.page).label} repository={state.repository} />
        <main id="main" tabIndex={-1} className="min-h-0 flex-1 overflow-y-auto">
          <CurrentPage />
        </main>
      </div>
    </div>
  );
}
