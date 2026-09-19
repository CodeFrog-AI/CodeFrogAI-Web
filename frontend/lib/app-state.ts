/**
 * The desktop shell's frontend state: which page is showing and which repository is selected.
 * Pure types and a reducer, so the logic is testable without React.
 */

export type Page = "repositories" | "agent" | "pull-requests" | "settings";

export interface PageDefinition {
  id: Page;
  label: string;
  description: string;
}

export const PAGES: readonly PageDefinition[] = [
  { id: "repositories", label: "Repositories", description: "Choose and inspect a repository" },
  { id: "agent", label: "Agent", description: "Ask CodeFrog about your code" },
  { id: "pull-requests", label: "Pull Requests", description: "Review changes on GitHub" },
  { id: "settings", label: "Settings", description: "AI provider and GitHub connection" },
];

export interface RepositoryInfo {
  name: string;
  path: string;
  branch: string;
  status: string;
  projectType: string;
  languages: readonly string[];
}

export interface AppState {
  page: Page;
  repository: RepositoryInfo | null;
}

export type AppAction =
  | { type: "navigate"; page: Page }
  | { type: "select-repository"; repository: RepositoryInfo }
  | { type: "clear-repository" };

export const initialState: AppState = { page: "repositories", repository: null };

export function isPage(value: unknown): value is Page {
  return PAGES.some((page) => page.id === value);
}

export function pageDefinition(page: Page): PageDefinition {
  return PAGES.find((definition) => definition.id === page) ?? PAGES[0];
}

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "navigate":
      return isPage(action.page) && action.page !== state.page ? { ...state, page: action.page } : state;
    case "select-repository":
      return { ...state, repository: action.repository };
    case "clear-repository":
      return state.repository === null ? state : { ...state, repository: null };
    default:
      return state;
  }
}
