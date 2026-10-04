"""RecentAgent — a from-scratch AI agent with RAG and tools.

Package layout:
    core.config      — provider presets and settings
    core.llm         — OpenAI-compatible LLM client (OpenAI / Groq / Ollama)
    core.embeddings  — neural (API) + TF-IDF fallback embedders
    core.rag         — chunking, knowledge base, vector search, persistence
    core.tools       — tool registry (calculator, web search, wikipedia, ...)
    core.agent       — ReAct reasoning loop that orchestrates tools
"""

__version__ = "1.0.0"
