"""LLM client.

One class, three providers: because OpenAI, Groq and Ollama all expose an
OpenAI-compatible `/v1/chat/completions` endpoint, we only need the official
`openai` Python package pointed at a different base_url. This keeps the
project dependency-light and makes the provider switchable at runtime.
"""
from __future__ import annotations

from openai import OpenAI

from .config import Settings


class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an error."""


class LLMClient:
    def __init__(self, settings: Settings):
        preset = settings.preset()
        self.model = settings.resolved_model()
        self.temperature = settings.temperature

        api_key = settings.resolved_api_key()
        if preset["needs_key"] and not api_key:
            raise LLMError(f"No API key configured for provider '{settings.provider}'.")

        kwargs: dict = {"timeout": 90.0}
        if preset["base_url"]:
            kwargs["base_url"] = preset["base_url"]
        # Ollama ignores the key but the client requires a non-empty one.
        self._client = OpenAI(api_key=api_key or "local", **kwargs)

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict]) -> str:
        """Send a chat completion request and return the assistant text."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as exc:  # network, auth, rate-limit, ...
            raise LLMError(f"LLM request failed ({type(exc).__name__}): {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        return (content or "").strip()

    # ------------------------------------------------------------------
    def test_connection(self) -> tuple[bool, str]:
        """Tiny health check used by the sidebar 'Test connection' button."""
        try:
            reply = self.chat(
                [{"role": "user", "content": "Reply with exactly the word: OK"}]
            )
        except LLMError as exc:
            return False, str(exc)
        return True, f"Connected to '{self.model}' — model replied: {reply[:60]!r}"
