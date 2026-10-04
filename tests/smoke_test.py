"""Offline smoke tests for RecentAgent — no API key or network required.

Run:  python tests/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from core.agent import ReActAgent, parse_react                 # noqa: E402
from core.embeddings import TfidfEmbedder                      # noqa: E402
from core.rag import KnowledgeBase, chunk_text                 # noqa: E402
from core.tools import build_default_tools, safe_eval, calculator_tool  # noqa: E402


def main() -> int:
    # ------------------------------------------------------------------ #
    # 1. Chunking
    text = " ".join(f"Sentence number {i} talks about retrieval augmented "
                    f"generation and agents." for i in range(120))
    chunks = chunk_text(text, size=500, overlap=100)
    assert 1 < len(chunks) < 50, f"unexpected chunk count {len(chunks)}"
    assert all(len(c) < 700 for c in chunks), "chunk too large"
    assert chunks[0][:80] != chunks[1][:80], "no forward progress between chunks"
    print("  PASS chunk_text -> %d chunks with overlap" % len(chunks))

    # ------------------------------------------------------------------ #
    # 2. Knowledge base + TF-IDF retrieval
    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(tmp, embedder=TfidfEmbedder())
        sample = (PROJECT / "sample_docs" / "02_rag_explained.txt").read_text()
        kb.add_document("02_rag_explained.txt", sample)
        kb.add_document("04_recent_agent_user_guide.txt",
                        (PROJECT / "sample_docs" / "04_recent_agent_user_guide.txt").read_text())
        hits = kb.search("what are the stages of the RAG pipeline?")
        assert hits, "retrieval returned nothing"
        assert hits[0]["source"] == "02_rag_explained.txt", \
            f"wrong top source: {hits[0]['source']}"
        hits2 = kb.search("how do I configure the ollama provider?")
        assert hits2 and hits2[0]["source"] == "04_recent_agent_user_guide.txt"
        kb.delete_document("04_recent_agent_user_guide.txt")
        print("  PASS KnowledgeBase: ingest, top-1 retrieval, delete")

        # reload from disk (persistence)
        kb2 = KnowledgeBase(tmp)
        kb2.set_embedder(TfidfEmbedder())
        kb2.sync()
        assert kb2.chunk_count() == kb.chunk_count() > 0, "persistence broken"
        print("  PASS persistence: vectors survive reload")

    # ------------------------------------------------------------------ #
    # 3. AST calculator (correctness + injection rejection)
    assert safe_eval("2 * (3 + 4) ** 2") == 98
    assert abs(safe_eval("sqrt(144) / 2") - 6.0) < 1e-9
    assert abs(safe_eval("pi * 2") - 6.283185307179586) < 1e-9
    print("  PASS safe_eval arithmetic")

    for payload in ["__import__('os').system('ls')",
                    "().__class__.__bases__",
                    "lambda: 1",
                    "os.getcwd"]:
        result = calculator_tool(payload)
        assert result.startswith("Error"), f"injection not blocked: {payload}"
    print("  PASS calculator blocks code-injection payloads")

    # ------------------------------------------------------------------ #
    # 4. ReAct parsing
    parsed = parse_react("Thought: I should calculate.\n"
                         "Action: calculator\nAction Input: 1 + 1")
    assert parsed["type"] == "action" and parsed["name"] == "calculator" \
        and parsed["input"] == "1 + 1", parsed
    parsed = parse_react("Thought: done.\nFinal Answer: The answer is 4.")
    assert parsed["type"] == "final" and parsed["answer"] == "The answer is 4."
    parsed = parse_react("Action: web_search\nAction Input: ```latest AI news```")
    assert parsed["input"] == "latest AI news", parsed
    print("  PASS ReAct output parsing (action / final / fenced input)")

    # ------------------------------------------------------------------ #
    # 5. Full agent loop with a scripted mock LLM (2 tool rounds + final)
    class FakeLLM:
        def __init__(self, script): self.script = list(script)
        def chat(self, messages): return self.script.pop(0)

    with tempfile.TemporaryDirectory() as tmp:
        kb = KnowledgeBase(tmp, embedder=TfidfEmbedder())
        kb.add_document("guide.txt",
                        (PROJECT / "sample_docs" / "04_recent_agent_user_guide.txt").read_text())
        tools, hits_sink = build_default_tools(kb)
        fake = FakeLLM([
            "Thought: I need math first.\nAction: calculator\nAction Input: 2 * (3 + 4) ** 2",
            "Thought: Now check the documents.\nAction: knowledge_base_search\n"
            "Action Input: what tools does the agent have?",
            "Thought: I have everything I need.\nFinal Answer: The result is 98, "
            "and the tools are described in guide.txt [source: guide.txt]",
        ])
        agent = ReActAgent(fake, tools, max_iterations=5)
        result = agent.run("test question")
        assert result.answer.startswith("The result is 98"), result.answer
        assert result.tool_rounds == 2 and len(result.steps) == 3
        assert "98" in result.steps[0]["observation"]
        assert hits_sink and hits_sink[0]["source"] == "guide.txt"
        print("  PASS agent loop: 2 tool rounds -> cited final answer")

    # ------------------------------------------------------------------ #
    # 6. datetime tool
    tools, _ = build_default_tools(None)
    out = tools["current_datetime"].func("")
    assert any(day in out for day in ("Monday", "Tuesday", "Wednesday", "Thursday",
                                      "Friday", "Saturday", "Sunday"))
    print("  PASS datetime tool")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
