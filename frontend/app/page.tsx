"use client";

import { useState } from "react";

export default function Home() {
  const [message, setMessage] = useState("");

  const checkBackend = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/");

      if (!response.ok) {
        throw new Error("Backend request failed");
      }

      const data = await response.json();
      setMessage(data.message);
    } catch (error) {
      console.error(error);
      setMessage("Could not connect to CodeFrog backend");
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      <h1 className="text-4xl font-bold">
        CodeFrog AI 🐸
      </h1>

      <p className="mt-4 text-gray-600">
        AI Software Engineer for GitHub
      </p>

      <button
        onClick={checkBackend}
        className="mt-8 rounded-lg bg-black px-6 py-3 text-white"
      >
        Check Backend
      </button>

      {message && (
        <p className="mt-6 text-lg">
          {message}
        </p>
      )}
    </main>
  );
}
