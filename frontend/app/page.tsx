import { AppShell } from "@/components/layout/AppShell";
import { AppProvider } from "@/lib/app-context";

export default function Home() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}
