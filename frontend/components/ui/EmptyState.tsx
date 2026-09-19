import type { ReactNode } from "react";

import { Icon, type IconName } from "@/components/ui/Icon";

interface EmptyStateProps {
  icon?: IconName;
  title: string;
  description: string;
  children?: ReactNode;
}

export function EmptyState({ icon = "frog", title, description, children }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border px-6 py-12 text-center">
      <span className="text-muted">
        <Icon name={icon} className="size-8" />
      </span>
      <h2 className="mt-4 text-base font-semibold">{title}</h2>
      <p className="mt-1 max-w-md text-sm text-muted">{description}</p>
      {children && <div className="mt-6 flex flex-wrap items-center justify-center gap-3">{children}</div>}
    </div>
  );
}
