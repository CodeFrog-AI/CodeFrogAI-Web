"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useApp } from "@/lib/app-context";
import { useAuth } from "@/lib/auth-context";
import { AUTH_ERROR_MESSAGES, completeSignIn, connectGitHub, type AuthErrorCode } from "@/lib/github-auth";

/**
 * Where the backend sends the browser after GitHub OAuth (the fragment holds a session token or an error code).
 * The token is read from the URL fragment, removed from the address bar immediately (inside
 * `completeSignIn`), and never rendered or logged.
 */
export default function AuthCallbackPage() {
  const router = useRouter();
  const app = useApp();
  const auth = useAuth();
  const started = useRef(false);
  const [failure, setFailure] = useState<AuthErrorCode | null>(null);

  useEffect(() => {
    // React may run this effect twice in development; the fragment can only be read once.
    if (started.current) return;
    started.current = true;

    void (async () => {
      const result = await completeSignIn(window.location.hash, {
        replaceUrl: () => window.history.replaceState(null, "", window.location.pathname),
      });
      if (result.status === "signed-in") {
        auth.setSignedIn(result.user);
        app.dispatch({ type: "navigate", page: "repositories" });
        router.replace("/");
      } else {
        setFailure(result.code);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="mx-auto flex min-h-full max-w-md items-center px-6 py-12">
      <Card title={failure ? "GitHub sign-in did not finish" : "Signing you in"} className="w-full">
        {failure ? (
          <div role="alert" className="space-y-4 text-sm">
            <p className="text-muted">{AUTH_ERROR_MESSAGES[failure]}</p>
            <div className="flex flex-wrap gap-3">
              <Button variant="primary" onClick={() => connectGitHub()}>
                Try again
              </Button>
              <Button onClick={() => router.replace("/")}>Back to CodeFrog</Button>
            </div>
          </div>
        ) : (
          <p role="status" className="text-sm text-muted">
            Finishing GitHub sign-in…
          </p>
        )}
      </Card>
    </main>
  );
}
