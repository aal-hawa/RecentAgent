"""Central configuration: provider presets and runtime settings.

Everything is overridable at runtime from the Streamlit sidebar, and
sensible defaults come from environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Provider presets. Every provider speaks the OpenAI-compatible protocol, so a
# single client implementation works for all of them — this is deliberate:
# the user can switch between cloud (OpenAI/Groq) and fully-offline (Ollama)
# backends without changing any application code.
# ---------------------------------------------------------------------------
PROVIDER_PRESETS: dict[str, dict] = {
    "ollama": {
        "label": "Ollama (100% offline, local)",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "needs_key": False,
        "supports_api_embeddings": True,
        "embedding_model": "nomic-embed-text",
        "hint": "Run: ollama pull llama3.1 && ollama pull nomic-embed-text",
    },
    "groq": {
        "label": "Groq (free API key)",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "needs_key": True,
        "supports_api_embeddings": False,  # Groq has no embeddings endpoint
        "embedding_model": None,
        "hint": "Get a free key at https://console.groq.com (no credit card).",
    },
    "google": {
        "label": "Google Gemini (free API key)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        # 2026-10: the "-latest" aliases (flash-lite-latest, flash-latest)
        # intermittently return empty completions on the free tier; the pinned
        # versioned models are healthy. fallback_model is used by LLMClient
        # when the primary keeps returning empty answers.
        "default_model": "gemini-3.5-flash",
        "fallback_model": "gemini-3.1-flash-lite",
        "needs_key": True,
        "supports_api_embeddings": False,  # no embeddings used -> TF-IDF fallback
        "embedding_model": None,
        "hint": "Free key at https://aistudio.google.com — set GEMINI_API or RECENT_API_KEY.",
    },
    "openai": {
        "label": "OpenAI (api.openai.com)",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "needs_key": True,
        "supports_api_embeddings": True,
        "embedding_model": "text-embedding-3-small",
        "hint": "Uses the same key for chat and embeddings.",
    },
}


@dataclass
class Settings:
    """Runtime settings for the LLM, embeddings and RAG pipeline."""

    provider: str = "ollama"        # ollama | groq | openai
    model: str = ""                 # "" -> preset default
    api_key: str = ""               # "" -> RECENT_API_KEY env var
    temperature: float = 0.2        # low temperature = more deterministic agent
    max_iterations: int = 8         # ReAct tool rounds per question
    chunk_size: int = 1000          # characters per RAG chunk
    chunk_overlap: int = 200        # character overlap between chunks
    top_k: int = 4                  # passages retrieved per query
    embedding_mode: str = "auto"    # auto | api | tfidf

    def preset(self) -> dict:
        return PROVIDER_PRESETS[self.provider]

    def resolved_model(self) -> str:
        return self.model or self.preset()["default_model"]

    def resolved_api_key(self) -> str:
        return (
            self.api_key
            or os.getenv("RECENT_API_KEY", "")
            or os.getenv("GEMINI_API", "")
        )

    def validate(self) -> list[str]:
        """Return a list of blocking configuration problems (empty = OK)."""
        problems: list[str] = []
        if self.provider not in PROVIDER_PRESETS:
            problems.append(f"Unknown provider: {self.provider}")
        if self.preset()["needs_key"] and not self.resolved_api_key():
            problems.append(
                f"No API key for {self.provider}. Enter it in the sidebar "
                f"or set RECENT_API_KEY in your .env file."
            )
        return problems
