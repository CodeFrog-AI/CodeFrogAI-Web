import { Icon } from "@/components/ui/Icon";
import { PAGES, type Page } from "@/lib/app-state";

interface SidebarProps {
  current: Page;
  onNavigate: (page: Page) => void;
}

/** Primary navigation. Below the `md` breakpoint it collapses to icons; the labels stay available to screen readers. */
export function Sidebar({ current, onNavigate }: SidebarProps) {
  return (
    <aside className="flex w-14 shrink-0 flex-col border-r border-border bg-surface md:w-56">
      <div className="flex h-14 items-center gap-2 border-b border-border px-4 font-semibold">
        <span className="text-accent">
          <Icon name="frog" />
        </span>
        <span className="hidden md:inline">CodeFrog</span>
      </div>
      <nav aria-label="Main" className="flex-1 p-2">
        <ul className="space-y-1">
          {PAGES.map((page) => {
            const active = page.id === current;
            return (
              <li key={page.id}>
                <button
                  type="button"
                  onClick={() => onNavigate(page.id)}
                  aria-current={active ? "page" : undefined}
                  title={page.label}
                  className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                    active ? "bg-surface-raised text-foreground" : "text-muted hover:bg-surface-raised hover:text-foreground"
                  }`}
                >
                  <Icon name={page.id} />
                  <span className="sr-only md:not-sr-only">{page.label}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
