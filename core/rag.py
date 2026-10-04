"""RAG (Retrieval-Augmented Generation) core.

Pipeline: ingest -> chunk -> embed -> index (in-memory numpy matrix) ->
retrieve by cosine similarity -> hand grounded passages + citations to the
agent. The knowledge base persists to disk (JSON + .npz) so uploaded
documents survive app restarts.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .embeddings import BaseEmbedder, TfidfEmbedder


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Sliding-window chunker over whitespace-normalized text.

    `size`   — target chunk length in characters
    `overlap`— characters carried over between consecutive chunks, so answers
               that straddle a boundary still retrieve complete context.

    A word-level sliding window is used instead of hard character cuts: it is
    simple, deterministic, and never splits a word in half. At demo scale
    (a few hundred chunks) it is fast and easy to reason about — a deliberate
    trade-off vs. semantic/sentence-aware splitters.
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []

    words = text.split(" ")
    chunks: list[str] = []
    start = 0
    while start < len(words):
        buf: list[str] = []
        length = 0
        j = start
        while j < len(words) and length < size:
            buf.append(words[j])
            length += len(words[j]) + 1
            j += 1
        chunks.append(" ".join(buf))
        if j >= len(words):
            break
        # rewind by ~`overlap` characters to create the overlap window
        back = j - 1
        carried = 0
        while back > start and carried < overlap:
            carried += len(words[back]) + 1
            back -= 1
        start = max(back + 1, start + 1)  # guarantee forward progress
    return chunks


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """Chunk store + vector index with disk persistence.

    Vectors are cached per chunk hash. Switching embedders (or refitting
    TF-IDF) invalidates the cache and re-embeds automatically, so the UI can
    swap providers freely without corrupting the index.
    """

    def __init__(self, persist_dir: str | Path, embedder: BaseEmbedder | None = None):
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or TfidfEmbedder()
        self.chunks: list[dict] = []          # {source, text, hash}
        self.vectors: list[np.ndarray | None] = []
        self._fingerprint: str = ""
        self.top_k: int = 4
        self.load()

    # ------------------------------------------------------------------ #
    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def set_embedder(self, embedder: BaseEmbedder) -> None:
        if embedder.fingerprint != self._fingerprint:
            self.vectors = [None] * len(self.chunks)
            self._fingerprint = embedder.fingerprint
        self.embedder = embedder

    def chunk_count(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ #
    def add_document(self, name: str, text: str, chunk_size: int = 1000,
                     chunk_overlap: int = 200) -> int:
        """Chunk + index a document. Returns the number of chunks added."""
        self.delete_document(name)  # re-ingestion replaces the old version
        pieces = chunk_text(text, size=chunk_size, overlap=chunk_overlap) or []
        for idx, piece in enumerate(pieces):
            self.chunks.append(
                {"source": name, "text": piece, "hash": _hash_text(piece), "chunk_id": idx}
            )
        self.vectors.extend([None] * len(pieces))
        self.sync()
        return len(pieces)

    def delete_document(self, name: str) -> None:
        keep = [(c, v) for c, v in zip(self.chunks, self.vectors) if c["source"] != name]
        self.chunks = [c for c, _ in keep]
        self.vectors = [v for _, v in keep]
        self.sync()

    def clear(self) -> None:
        self.chunks, self.vectors = [], []
        self.sync()

    def list_documents(self) -> list[dict]:
        """Aggregate chunk stats per document for the sidebar."""
        stats: dict[str, dict] = {}
        for c in self.chunks:
            entry = stats.setdefault(c["source"], {"name": c["source"], "chunks": 0, "chars": 0})
            entry["chunks"] += 1
            entry["chars"] += len(c["text"])
        return sorted(stats.values(), key=lambda d: d["name"].lower())

    # ------------------------------------------------------------------ #
    def sync(self) -> None:
        """Make vectors consistent with chunks (embed anything missing)."""
        if not self.chunks:
            self._save()
            return
        # TF-IDF's embedding space depends on the whole corpus -> always refit.
        if self.embedder.kind == "tfidf":
            self.vectors = [None] * len(self.chunks)
        self.embedder.fit([c["text"] for c in self.chunks])
        missing = [i for i, v in enumerate(self.vectors) if v is None]
        if missing:
            matrix = self.embedder.encode([self.chunks[i]["text"] for i in missing])
            for row, i in zip(matrix, missing):
                self.vectors[i] = row
        self._save()

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Top-k semantic search. Returns [{source, chunk_id, text, score}]."""
        if not self.chunks:
            return []
        k = k or self.top_k
        q = np.asarray(self.embedder.encode_query(query), dtype="float32")

        matrix = np.stack([v for v in self.vectors if v is not None])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        q /= np.linalg.norm(q) + 1e-10
        scores = matrix @ q  # cosine similarity (rows are unit-normalized)

        order = np.argsort(-scores)[:k]
        live = [i for i, v in enumerate(self.vectors) if v is not None]
        hits = []
        for pos in order:
            idx = live[int(pos)]
            c = self.chunks[idx]
            hits.append(
                {"source": c["source"], "chunk_id": c["chunk_id"],
                 "text": c["text"], "score": float(scores[pos])}
            )
        return hits

    # ------------------------------------------------------------------ #
    # Persistence: chunks.json (portable) + vectors.npz (binary matrix).
    def _save(self) -> None:
        (self.dir / "chunks.json").write_text(
            json.dumps({"fingerprint": self._fingerprint, "chunks": self.chunks},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        present = [v for v in self.vectors if v is not None]
        if present:
            np.savez(self.dir / "vectors.npz",
                     matrix=np.stack(present).astype("float32"),
                     rows=np.array([i for i, v in enumerate(self.vectors) if v is not None]))
        else:
            (self.dir / "vectors.npz").unlink(missing_ok=True)

    def load(self) -> None:
        chunks_file = self.dir / "chunks.json"
        if not chunks_file.exists():
            return
        try:
            data = json.loads(chunks_file.read_text(encoding="utf-8"))
            self._fingerprint = data.get("fingerprint", "")
            self.chunks = data.get("chunks", [])
        except (json.JSONDecodeError, OSError):
            self.chunks, self._fingerprint = [], ""
        self.vectors = [None] * len(self.chunks)

        vec_file = self.dir / "vectors.npz"
        if vec_file.exists() and self.chunks:
            try:
                blob = np.load(vec_file)
                if blob["matrix"].shape[1:] == ():  # guard against dim mismatch
                    raise ValueError
                for row, i in zip(blob["matrix"], blob["rows"]):
                    if int(i) < len(self.vectors):
                        self.vectors[int(i)] = row.astype("float32")
            except (OSError, KeyError, ValueError):
                self.vectors = [None] * len(self.chunks)
