import { useId, type ReactNode } from "react";

interface CardProps {
  title?: string;
  description?: string;
  children?: ReactNode;
  className?: string;
}

/** A bordered surface. When it has a title it is a labelled region for assistive technology. */
export function Card({ title, description, children, className = "" }: CardProps) {
  const headingId = useId();
  return (
    <section
      aria-labelledby={title ? headingId : undefined}
      className={`rounded-lg border border-border bg-surface p-5 ${className}`}
    >
      {title && (
        <h2 id={headingId} className="text-sm font-semibold">
          {title}
        </h2>
      )}
      {description && <p className="mt-1 text-sm text-muted">{description}</p>}
      {children && <div className={title || description ? "mt-4" : ""}>{children}</div>}
    </section>
  );
}
