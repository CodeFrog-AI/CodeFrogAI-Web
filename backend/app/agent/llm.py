"""LLM provider abstraction and an OpenAI-compatible chat-completions implementation.

Messages use the OpenAI chat format (`system`/`user`/`assistant`/`tool` roles), which most
compatible providers accept, so another provider only needs to implement `LLMProvider`.
"""

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.core.config import get_settings

REQUEST_TIMEOUT_SECONDS = 60.0

Message = dict[str, Any]
ToolSpec = dict[str, Any]  # {"name": str, "description": str, "parameters": <JSON schema>}


class LLMError(RuntimeError):
    """An LLM request failed; messages never include keys, prompts, or provider response bodies."""


class LLMNotConfiguredError(LLMError):
    """No LLM API key is configured."""


@dataclass(frozen=True)
class ToolCallRequest:
    """A tool call requested by the model. `arguments` is the raw JSON text it produced."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest]


class LLMProvider(Protocol):
    model: str

    def complete(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        """Return the model's next message. With `tools` omitted the model cannot call tools."""
        ...


class OpenAICompatibleLLM:
    """Call an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    def complete(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = [{"type": "function", "function": spec} for spec in tools]
            payload["tool_choice"] = "auto"
        try:
            with httpx.Client(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = client.post("/chat/completions", json=payload)
        except httpx.HTTPError:
            raise LLMError("LLM request failed") from None
        if response.status_code >= 400:
            raise LLMError("LLM provider rejected the request")
        return _parse_response(response)


def _parse_response(response: httpx.Response) -> LLMResponse:
    try:
        message = response.json()["choices"][0]["message"]
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise TypeError
        calls = []
        for raw in message.get("tool_calls") or []:
            arguments = raw["function"].get("arguments") or ""
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            calls.append(ToolCallRequest(id=str(raw["id"]), name=str(raw["function"]["name"]), arguments=arguments))
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise LLMError("LLM provider returned an invalid response") from None
    return LLMResponse(content=content, tool_calls=calls)


def get_llm_provider() -> LLMProvider:
    """Build the configured provider, or raise if no API key is set."""

    settings = get_settings()
    if settings.llm_api_key is None:
        raise LLMNotConfiguredError("LLM provider is not configured")
    return OpenAICompatibleLLM(settings.llm_api_key.get_secret_value(), settings.llm_model, settings.llm_base_url)
