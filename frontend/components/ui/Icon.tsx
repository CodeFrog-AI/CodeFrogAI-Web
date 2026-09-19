import type { Page } from "@/lib/app-state";

export type IconName = Page | "frog";

const PATHS: Record<IconName, string> = {
  repositories: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z",
  agent: "M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12Z",
  "pull-requests": "M6 3v12m0 0a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm0-9a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm0-6V9a3 3 0 0 0-3-3h-3",
  settings: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 0 0-2-1.2L14.5 3h-4l-.4 2.6a7.6 7.6 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7.6 7.6 0 0 0 2 1.2l.4 2.6h4l.4-2.6a7.6 7.6 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2Z",
  frog: "M6 9a3 3 0 1 1 5-2.2h2A3 3 0 1 1 18 9c1.8.9 3 2.7 3 4.8C21 17.2 17 19 12 19s-9-1.8-9-5.2C3 11.7 4.2 9.9 6 9Z",
};

/** A decorative icon. Buttons that use it must carry their own text or aria-label. */
export function Icon({ name, className = "size-5" }: { name: IconName; className?: string }) {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
