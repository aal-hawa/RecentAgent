"""Real end-to-end test against a local Ollama server — no cloud, no API keys.

Run AFTER you have Ollama installed and at least one instruct model pulled:

    python tests/real_ollama_test.py                 # auto-pick models
    python tests/real_ollama_test.py llama3.1        # force a chat model

What it verifies:
  1. Ollama is reachable and the chat model responds.
  2. Neural embeddings work when nomic-embed-text is present
     (otherwise the pipeline gracefully degrades to TF-IDF).
  3. RAG ingestion + retrieval return the right document.
  4. The ReAct agent answers a knowledge question WITH citations.
  5. The ReAct agent uses the calculator tool and gets the math right.

Exit code 0 = all stages passed.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

# Make `core` importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent import ReActAgent  # noqa: E402
from core.config import Settings  # noqa: E402
from core.embeddings import build_embedder  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.rag import KnowledgeBase  # noqa: E402
from core.tools import build_default_tools  # noqa: E402

OLLAMA_BASE = "http://localhost:11434"
SAMPLE_DOC = Path(__file__).resolve().parents[1] / "sample_docs" / "02_rag_explained.txt"

PASSED: list[str] = []
FAILED: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def main() -> int:
    print("=" * 64)
    print("RecentAgent — REAL end-to-end test on local Ollama")
    print("=" * 64)

    models = ollama_models()
    if not models:
        print("Ollama not reachable on", OLLAMA_BASE)
        print("Start it with:  ollama serve   (or run the Ollama app)")
        return 1
    print(f"Ollama models found: {', '.join(models)}")

    # ---- pick models -------------------------------------------------
    forced = sys.argv[1] if len(sys.argv) > 1 else ""
    chat_model = forced or next(
        (m for m in models if any(k in m for k in ("llama3.1", "llama3.2", "qwen2.5", "mistral", "granite"))),
        models[0],
    )
    has_embed = any("nomic-embed-text" in m for m in models)
    settings = Settings(
        provider="ollama",
        model=chat_model,
        temperature=0.2,
        max_iterations=6,
        embedding_mode="auto",
    )
    print(f"Chat model : {settings.resolved_model()}")
    print(f"Embeddings : {'nomic-embed-text' if has_embed else 'TF-IDF fallback (nomic-embed-text not pulled)'}")
    print()

    # ---- 1. LLM connection ------------------------------------------
    print("[1/5] LLM connection")
    llm = LLMClient(settings)
    try:
        ok, msg = llm.test_connection()
        report("chat model responds", ok, msg)
        if not ok:
            return 1
    except Exception as exc:
        report("chat model responds", False, f"{type(exc).__name__}: {exc}")
        return 1

    # ---- 2. Embeddings ----------------------------------------------
    print("[2/5] Embeddings")
    embedder, notice = build_embedder(settings)
    vecs = embedder.encode(["hello world", "retrieval augmented generation"])
    report(
        "embedder produces vectors",
        vecs.shape[0] == 2 and vecs.shape[1] > 0,
        f"dims={vecs.shape[1]} mode={type(embedder).__name__}",
    )
    if notice:
        print(f"  notice: {notice}")

    # ---- 3. RAG ingestion + retrieval --------------------------------
    print("[3/5] RAG ingestion + retrieval")
    tmp = Path(tempfile.mkdtemp(prefix="recentagent-test-"))
    try:
        kb = KnowledgeBase(tmp, embedder=embedder)
        kb.add_document(SAMPLE_DOC.name, SAMPLE_DOC.read_text(encoding="utf-8"))
        report("document ingested", kb.chunk_count() > 0, f"{kb.chunk_count()} chunks")

        hits = kb.search("How does RAG reduce hallucinations?")
        report(
            "retrieval finds the right doc",
            bool(hits) and "rag" in hits[0]["source"].lower(),
            f"top hit: {hits[0]['source']} (score {hits[0]['score']:.3f})" if hits else "no hits",
        )

        # ---- 4. ReAct agent: knowledge question ----------------------
        print("[4/5] ReAct agent — knowledge question (RAG, with citations)")
        tools, hits_sink = build_default_tools(kb)
        agent = ReActAgent(llm, tools, max_iterations=settings.max_iterations)
        result = agent.run(
            "According to the documents, what is Retrieval-Augmented Generation "
            "and how does it reduce hallucinations? Use the knowledge base and cite sources."
        )
        for i, s in enumerate(result.steps, 1):
            kind = s.get("kind", "?")
            if kind == "action":
                print(f"    step {i}: Action {s.get('action')}({str(s.get('action_input'))[:60]!r})")
            elif kind == "observation":
                print(f"    step {i}: Observation {str(s.get('observation'))[:80]!r}")
            elif kind == "final":
                print(f"    step {i}: Final Answer reached")
        answer_l = result.answer.lower()
        report("agent produced a final answer", bool(result.answer.strip()),
               f"{len(result.answer)} chars, {result.tool_rounds} tool rounds")
        report("answer mentions RAG/retrieval", "retriev" in answer_l or "rag" in answer_l)
        report("answer cites documents", "[source:" in answer_l or bool(result.sources))

        # ---- 5. ReAct agent: calculator tool -------------------------
        print("[5/5] ReAct agent — calculator tool (17 * 23 + sqrt(144))")
        tools2, _ = build_default_tools(None)
        agent2 = ReActAgent(llm, tools2, max_iterations=settings.max_iterations)
        result2 = agent2.run("Compute 17 * 23 + sqrt(144) exactly. Use the calculator tool.")
        print(f"    answer: {result2.answer.strip()[:120]}")
        report("calculator used", any(s.get("action") == "calculator" for s in result2.steps))
        report("result correct (403)", "403" in result2.answer)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- summary -----------------------------------------------------
    print()
    print("=" * 64)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed stages:", ", ".join(FAILED))
        return 1
    print("All stages passed — RecentAgent is fully functional on Ollama.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
