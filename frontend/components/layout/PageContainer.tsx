import type { ReactNode } from "react";

interface PageContainerProps {
  title: string;
  description?: string;
  children: ReactNode;
}

/** The standard page body: one h1, then the page's content, at a comfortable reading width. */
export function PageContainer({ title, description, children }: PageContainerProps) {
  return (
    <div className="mx-auto w-full max-w-4xl px-6 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {description && <p className="mt-1 text-sm text-muted">{description}</p>}
      <div className="mt-6 space-y-6">{children}</div>
    </div>
  );
}
