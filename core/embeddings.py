"""Embedding backends with graceful degradation.

Two interchangeable implementations behind one interface:

1. ApiEmbedder   — neural embeddings via any OpenAI-compatible /v1/embeddings
                   endpoint (OpenAI, Ollama). Best semantic quality.
2. TfidfEmbedder — classic TF-IDF vectors (scikit-learn). Zero network calls,
                   zero model downloads, always works — even fully offline.

The RAG layer treats them identically, which is a deliberate design point:
the retrieval backend can degrade from neural to lexical without any code
change (useful when no embeddings API is available, e.g. Groq, or when
running on a machine with no model downloads allowed).
"""
from __future__ import annotations

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings


class BaseEmbedder:
    """Interface every embedder implements."""

    kind: str = "base"
    fingerprint: str = "base"  # identifies the embedding space for the cache

    def fit(self, texts: list[str]) -> None:
        """Prepare the embedder for the current corpus (TF-IDF refits, API no-op)."""

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
class ApiEmbedder(BaseEmbedder):
    """Neural embeddings from an OpenAI-compatible endpoint."""

    kind = "api"

    def __init__(self, settings: Settings, model: str):
        preset = settings.preset()
        self.model = model
        self._base = preset["base_url"] or "https://api.openai.com/v1"
        self.fingerprint = f"api:{self._base}:{model}"
        self._client = OpenAI(
            api_key=settings.resolved_api_key() or "local",
            base_url=self._base,
            timeout=60.0,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        batch_size = 64  # stay well inside per-request token limits
        for start in range(0, len(texts), batch_size):
            batch = [t[:8000] for t in texts[start : start + batch_size]]
            response = self._client.embeddings.create(model=self.model, input=batch)
            data = sorted(response.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in data)
        return np.asarray(vectors, dtype="float32")


# ---------------------------------------------------------------------------
class TfidfEmbedder(BaseEmbedder):
    """Lexical TF-IDF vectors — the zero-dependency fallback."""

    kind = "tfidf"
    fingerprint = "tfidf"

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",  # rows come out unit-length -> dot product == cosine
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        if texts:
            # refit on the full corpus — the vocabulary must cover every chunk
            self._vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
                norm="l2",
            )
            self._vectorizer.fit(texts)
            self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("TfidfEmbedder must be fitted on a non-empty corpus first.")
        return self._vectorizer.transform(texts).toarray().astype("float32")


# ---------------------------------------------------------------------------
def build_embedder(settings: Settings) -> tuple[BaseEmbedder, str | None]:
    """Choose the best available embedder.

    Returns (embedder, notice). `notice` is a user-facing warning string when
    we had to degrade to TF-IDF, otherwise None.
    """
    preset = settings.preset()
    mode = settings.embedding_mode

    if mode == "tfidf":
        return TfidfEmbedder(), "Embedding mode forced to TF-IDF (lexical retrieval)."

    if mode == "api" or (mode == "auto" and preset.get("supports_api_embeddings")):
        embedder = ApiEmbedder(settings, preset["embedding_model"])
        try:
            # Cheap probe so a broken endpoint falls back *before* ingestion.
            embedder.encode(["connection probe"])
            return embedder, None
        except Exception as exc:
            notice = (
                f"Neural embeddings unavailable ({type(exc).__name__}: {exc}) — "
                "falling back to TF-IDF lexical retrieval."
            )
            return TfidfEmbedder(), notice

    return TfidfEmbedder(), (
        f"'{settings.provider}' has no embeddings endpoint — using built-in "
        "TF-IDF lexical retrieval (works great for demos)."
    )
