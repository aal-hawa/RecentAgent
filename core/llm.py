"""LLM client.

One class, three providers: because OpenAI, Groq and Ollama all expose an
OpenAI-compatible `/v1/chat/completions` endpoint, we only need the official
`openai` Python package pointed at a different base_url. This keeps the
project dependency-light and makes the provider switchable at runtime.
"""
from __future__ import annotations

import time

from openai import OpenAI

from .config import Settings


class LLMError(RuntimeError):
    """Raised when the LLM backend is unreachable or returns an error."""


class LLMClient:
    def __init__(self, settings: Settings):
        preset = settings.preset()
        self.model = settings.resolved_model()
        self.temperature = settings.temperature
        # Optional second model used when the primary keeps returning empty
        # completions (the Gemini free tier's "-latest" aliases did exactly
        # that in 2026-10). Only set for providers that have one.
        self.fallback_model = preset.get("fallback_model")
        if self.fallback_model == self.model:
            self.fallback_model = None

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
        """Send a chat completion request and return the assistant text.

        Empty completions (HTTP 200, zero completion tokens) happen on some
        free tiers — most notoriously the Gemini "-latest" aliases — so each
        model gets a couple of quick retries, and providers that define a
        fallback_model switch to it before giving up.
        """
        models = [self.model] + ([self.fallback_model] if self.fallback_model else [])
        attempts_per_model = 3 if self.fallback_model else 4
        last_detail = "no attempt made"
        for model in models:
            for _attempt in range(attempts_per_model):
                try:
                    response = self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                    )
                except Exception as exc:  # network, auth, rate-limit, ...
                    raise LLMError(f"LLM request failed ({type(exc).__name__}): {exc}") from exc

                content = response.choices[0].message.content if response.choices else None
                content = (content or "").strip()
                if content:
                    return content
                last_detail = "empty completion (200 OK, 0 tokens)"
                time.sleep(0.8)  # empty responses arrive fast; retry briefly
            # primary produced repeated empty completions -> try the fallback
        raise LLMError(
            "The model returned empty answers repeatedly "
            f"(tried: {', '.join(models)}; {last_detail}). "
            "This is a known free-tier hiccup — please try again."
        )

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
