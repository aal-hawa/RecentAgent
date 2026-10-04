"""RecentAgent — AI Assistant with RAG & Tools (Streamlit UI).

Run:  streamlit run app.py   (or double-click start.bat on Windows)
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.agent import ReActAgent
from core.config import PROVIDER_PRESETS, Settings
from core.embeddings import build_embedder
from core.llm import LLMClient, LLMError
from core.rag import KnowledgeBase
from core.tools import build_default_tools

load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parent
KB_DIR = PROJECT_DIR / "data" / "kb"
SAMPLES_DIR = PROJECT_DIR / "sample_docs"

st.set_page_config(page_title="RecentAgent — AI Agent with RAG", page_icon="🤖",
                   layout="wide")
st.title("🤖 RecentAgent")
st.caption("An AI agent with **RAG** (document knowledge base) and **tools** "
           "(calculator, web search, Wikipedia) — built from scratch with an "
           "inspectable ReAct reasoning loop.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("samples_loaded", False)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
def ollama_installed_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Query a local Ollama server for installed model names ([] if unreachable)."""
    try:
        import json as _json
        import urllib.request as _req
        with _req.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            data = _json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase(KB_DIR)


@st.cache_resource(show_spinner=False)
def get_llm(provider: str, model: str, api_key: str, temperature: float) -> LLMClient:
    return LLMClient(Settings(provider=provider, model=model, api_key=api_key,
                              temperature=temperature))


@st.cache_resource(show_spinner=False)
def get_embedder(provider: str, api_key: str, mode: str):
    return build_embedder(Settings(provider=provider, api_key=api_key,
                                   embedding_mode=mode))


def extract_text(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded file (txt/md/pdf/docx)."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ("txt", "md", "markdown"):
        return raw.decode("utf-8", errors="ignore")
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == "docx":
        import docx
        document = docx.Document(io.BytesIO(raw))
        return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    raise ValueError(f"Unsupported file type: .{ext}")


def load_sample_documents() -> int:
    kb = get_kb()
    total = 0
    for path in sorted(SAMPLES_DIR.glob("*.txt")):
        total += kb.add_document(path.name, path.read_text(encoding="utf-8"))
    return total


# ---------------------------------------------------------------------------
# Sidebar — configuration + knowledge base management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox(
        "LLM provider",
        list(PROVIDER_PRESETS),
        format_func=lambda p: PROVIDER_PRESETS[p]["label"],
        index=0,
    )
    preset = PROVIDER_PRESETS[provider]
    env_model = os.getenv("RECENT_MODEL", "")
    # Model picker. For Ollama we show a DROPDOWN populated from the locally
    # installed models (`ollama list` / /api/tags) so it's impossible to pick
    # a model that isn't pulled — no more 404s from typos or missing pulls.
    if provider == "ollama":
        installed = ollama_installed_models()
        chat_models = [m for m in installed if "embed" not in m]
        options = chat_models or installed
        if options:
            default_index = options.index(env_model) if env_model in options else 0
            model = st.selectbox(
                "Model (installed locally)",
                options,
                index=default_index,
                key=f"model_{provider}",
                help="Populated from your local Ollama server. Pull more with: ollama pull <model>",
            )
            st.caption(f"{len(installed)} model(s) installed — pull more with: ollama pull <model>")
        else:
            st.warning("No Ollama models found. Pull one first, e.g.: ollama pull llama3.1")
            model = st.text_input("Model", value=env_model or preset["default_model"],
                                  key=f"model_text_{provider}")
    else:
        model = st.text_input("Model", value=env_model or preset["default_model"],
                              key=f"model_{provider}")
    api_key = "" if provider == "ollama" else st.text_input(
        "API key", value=os.getenv("RECENT_API_KEY", ""), type="password"
    )
    st.caption(preset["hint"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05,
                            help="Low = focused and deterministic (best for agents).")
    max_iterations = st.slider("Max tool rounds per question", 2, 12, 8)

    st.divider()
    st.subheader("🛠️ Agent settings")
    enable_net = st.checkbox(
        "🌐 Internet tools (web_search + wikipedia)", value=True,
        help="OFF = the agent runs 100% offline: only knowledge_base_search, "
             "calculator and current_datetime are available.")
    top_k = st.slider("Retrieved passages per query (top_k)", 2, 8, 4,
                      help="How many knowledge-base chunks the agent sees per search.")
    emb_mode = st.selectbox(
        "Embedding mode", ["auto", "api", "tfidf"],
        help="auto = neural embeddings when the provider supports them, "
             "TF-IDF fallback otherwise. tfidf = always lexical (zero network).")

    settings = Settings(provider=provider, model=model, api_key=api_key,
                        temperature=temperature, max_iterations=max_iterations,
                        top_k=top_k, embedding_mode=emb_mode)
    problems = settings.validate()

    if st.button("🔌 Test connection", use_container_width=True, disabled=bool(problems)):
        ok, message = get_llm(provider, model, api_key, temperature).test_connection()
        (st.success if ok else st.error)(message)

    st.divider()
    st.header("📚 Knowledge Base (RAG)")

    embedder, notice = get_embedder(provider, api_key, emb_mode)
    if notice:
        st.info(notice)

    kb = get_kb()
    kb.set_embedder(embedder)
    st.caption(f"📚 KB: {len(kb.list_documents())} doc(s) · {kb.chunk_count()} chunks · "
               f"embedder: {type(embedder).__name__}")

    uploads = st.file_uploader("Upload documents (PDF, TXT, MD, DOCX)",
                               type=["pdf", "txt", "md", "docx"],
                               accept_multiple_files=True)
    if uploads and st.button("➕ Ingest into knowledge base", use_container_width=True):
        added = 0
        for upload in uploads:
            try:
                text = extract_text(upload.name, upload.getvalue())
                if not text.strip():
                    st.warning(f"'{upload.name}' had no extractable text "
                               "(scanned PDFs need OCR).")
                    continue
                n = kb.add_document(upload.name, text)
                added += n
                st.toast(f"Ingested {upload.name} ({n} chunks)", icon="✅")
            except Exception as exc:
                st.error(f"Failed to read {upload.name}: {exc}")
        if added:
            st.rerun()

    with st.expander("📝 Or paste raw text"):
        paste_name = st.text_input("Document name", value="pasted_notes.txt")
        paste_text = st.text_area("Text", height=140)
        if st.button("Add pasted text") and paste_text.strip():
            kb.add_document(paste_name or "pasted_notes.txt", paste_text)
            st.toast("Pasted text ingested", icon="✅")
            st.rerun()

    if st.button("📦 Load sample documents", use_container_width=True):
        n = load_sample_documents()
        st.session_state["samples_loaded"] = True
        st.toast(f"Loaded sample documents ({n} chunks)", icon="📦")
        st.rerun()

    docs = kb.list_documents()
    if docs:
        st.markdown(f"**{len(docs)} document(s)** · {kb.chunk_count()} chunks indexed")
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"`{doc['name']}` — {doc['chunks']} chunks")
            if col2.button("🗑️", key=f"del_{doc['name']}",
                           help=f"Remove {doc['name']}"):
                kb.delete_document(doc["name"])
                st.rerun()
        if st.button("🧹 Clear entire knowledge base", use_container_width=True):
            kb.clear()
            st.rerun()
    else:
        st.caption("Empty — upload files or load the sample documents.")

    st.divider()
    with st.expander("ℹ️ About this project"):
        st.markdown(
            "**RecentAgent** demonstrates a production-shaped AI agent:\n\n"
            "1. **RAG** — upload → chunk → embed → vector search → cited answers\n"
            "2. **ReAct agent** — Thought → Action → Observation loop (from scratch)\n"
            "3. **Tools** — knowledge base, calculator (AST-sandboxed), web search, Wikipedia\n"
            "4. **Providers** — Ollama / Groq / OpenAI behind one OpenAI-compatible client\n\n"
            "See `README.md` in the repository."
        )

# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------
def render_steps(steps: list[dict]) -> None:
    """Show the ReAct trace — every reasoning step, fully transparent."""
    with st.expander("🧠 Agent reasoning trace (ReAct loop)"):
        for step in steps:
            st.markdown(f"**Round {step['iteration']}**")
            if step.get("thought"):
                st.markdown(f"> 💭 **Thought:** {step['thought']}")
            if step.get("action"):
                st.markdown(
                    f"⚡ **Action:** `{step['action']}`  \n"
                    f"📥 **Action Input:** `{step['action_input']}`"
                )
                obs = step.get("observation", "")
                st.markdown("📤 **Observation:**")
                st.text(obs[:1200] + (" …[truncated]" if len(obs) > 1200 else ""))
            if step.get("final") and not step.get("action"):
                st.markdown("🏁 _Final answer produced._")
            st.divider()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources used ({len(sources)} passages)"):
        for i, hit in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] `{hit['source']}`** — chunk #{hit['chunk_id']} · "
                f"similarity `{hit['score']:.2f}`"
            )
            st.text(hit["text"][:600] + ("…" if len(hit["text"]) > 600 else ""))
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_steps(message.get("steps", []))
            render_sources(message.get("sources", []))

# ---------------------------------------------------------------------------
# Auto-load samples on very first launch so the demo works instantly
# ---------------------------------------------------------------------------
if not kb.list_documents() and not st.session_state["samples_loaded"]:
    load_sample_documents()
    st.session_state["samples_loaded"] = True
    st.toast("Sample documents loaded — try asking: "
             "'What is RAG and why is it useful?'", icon="📦")
    st.rerun()

# ---------------------------------------------------------------------------
# Chat input -> agent
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask anything, or ask about your documents…"):
    if problems:
        st.error(" ".join(problems))
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            llm = get_llm(provider, model, api_key, temperature)
            tools, hits_sink = build_default_tools(kb, include_net=enable_net,
                                                   top_k=top_k)
            agent = ReActAgent(llm, tools, max_iterations=max_iterations)
            history = [(m["role"], m["content"])
                       for m in st.session_state.messages[:-1]]
            with st.spinner("Thinking… (watch the reasoning trace after answering)"):
                result = agent.run(prompt, history)
        except LLMError as exc:
            st.error(f"**LLM error:** {exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ LLM error: {exc}"})
            st.stop()

        result.sources = [dict(h) for h in hits_sink]
        st.markdown(result.answer)
        render_steps(result.steps)
        render_sources(result.sources)
        st.session_state.messages.append({
            "role": "assistant", "content": result.answer,
            "steps": result.steps, "sources": result.sources,
        })
