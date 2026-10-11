"""RecentAgent — hosted web demo (TESTING ONLY, not the product).

A faithful, dependency-light port of the local Streamlit app's core:
the same ReAct loop (Thought -> Action -> Observation -> Final Answer),
the same tool philosophy, and a TF-IDF knowledge base over the same sample
documents. It exists so the agent behavior can be tried in a browser.

Differences from the local app (by design):
  - LLM: Google Gemini (gemini-flash-latest) via the OpenAI-compatible
    endpoint, key provided as the GEMINI_API environment variable
  - tools: calculator (same AST sandbox), wikipedia (public REST API),
    knowledge_base_search (TF-IDF, from scratch), current_datetime
  - web_search (DuckDuckGo) is NOT included: it is unreliable from
    serverless IPs and needs an extra dependency
  - conversation history and the knowledge base reset on cold starts

The full app runs locally with Streamlit (see the repository README).
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

app = FastAPI(title="RecentAgent demo")

BASE_DIR = Path(__file__).parent
GEMINI_BASE = os.environ.get("RECENT_DEMO_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
# flash-lite-latest: the reliable free-tier choice as of 2026-10 — the
# full-size "flash-latest" alias currently routes to a degraded backend
# (multi-second to minute-long stalls, 429/503 under load).
MODEL = os.environ.get("RECENT_MODEL", "gemini-flash-lite-latest")
MAX_ITERATIONS = 4          # tool rounds per question (demo keeps answers fast)
WALL_BUDGET_S = 48          # hard stop before the 60s function limit
TOOL_OBS_CAP = 2500         # context hygiene, same as the local app


# ---------------------------------------------------------------------------
# LLM client — OpenAI-compatible endpoint, stdlib only
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    pass


def llm_chat(messages: list[dict], max_tokens: int = 1800, timeout: float = 25.0,
             attempts: int = 3, deadline: float | None = None,
             model: str | None = None) -> tuple[str, dict]:
    """One chat completion with small retries on transient free-tier errors.

    Returns (content, full assistant message dict). The full message includes
    Gemini's `extra_content.google.thought_signature`, which MUST be sent
    back in later turns for thinking models — dropping it makes some serving
    paths return empty completions.

    The free Gemini tier also occasionally answers 429 (quota) / 5xx
    (overloaded) or stalls; retrying once or twice inside the caller's
    deadline is usually enough. `deadline` is a time.monotonic() timestamp —
    attempts are only started if they fit before it.
    """
    key = os.environ.get("GEMINI_API") or os.environ.get("GOOGLE_API_KEY") or ""
    if not key:
        raise LLMError("GEMINI_API is not configured on the server.")
    body = json.dumps({
        "model": model or MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    last_detail = "no attempt made"
    for attempt in range(1, attempts + 1):
        if deadline is not None and attempt > 1 and time.monotonic() + timeout + 1.5 > deadline:
            break  # not enough time left for another try
        req = urllib.request.Request(
            GEMINI_BASE.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            last_detail = f"HTTP {exc.code}: " + exc.read().decode("utf-8", "replace")[:200]
            if exc.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(2.5 * attempt)  # transient quota/overload — back off, retry
                continue
            raise LLMError(f"LLM request failed ({last_detail})") from exc
        except Exception as exc:  # timeout, network, ...
            last_detail = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(2.0)
                continue
            raise LLMError(f"LLM request failed ({last_detail})") from exc

        # HTTP 200 — parse the body. Note: Gemini thinking models sometimes
        # return 200 with an EMPTY message (completion_tokens: 0, no content
        # key). That is transient — retry instead of failing the answer.
        try:
            data = json.loads(raw)
            message = data["choices"][0]["message"]
        except ValueError:
            last_detail = f"invalid JSON: {raw[:120]!r}"
            if attempt < attempts:
                time.sleep(1.5)
                continue
            raise LLMError(f"LLM returned invalid JSON ({last_detail})")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected LLM response shape: {raw[:200]!r}") from exc
        content = (message.get("content") or "").strip()
        if content:
            return content, message
        # Empty completions are how the free tier soft-throttles under load —
        # retrying too aggressively burns quota, so we back off meaningfully
        # and eventually surface a clear, honest message to the visitor.
        last_detail = f"empty content (raw: {raw[:200]!r})"
        if attempt < attempts:
            time.sleep(3.0 * attempt)
            continue
        raise LLMError(
            "The Gemini free tier is busy or rate-limited right now "
            f"({last_detail[:60]}) — please try again in a moment."
        )
    raise LLMError(f"LLM unavailable after {attempts} attempts ({last_detail})")


# ---------------------------------------------------------------------------
# Tool 1 — calculator (AST sandbox, identical rules to the local app)
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "log": math.log, "log2": math.log2,
    "log10": math.log10, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "round": round,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    """Evaluate pure arithmetic WITHOUT eval() — whitelisted AST nodes only."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1_000:
                raise ValueError("Exponent too large (max 1000).")
            return _ALLOWED_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _ALLOWED_FUNCS and len(node.args) == 1
                and not node.keywords):
            return _ALLOWED_FUNCS[node.func.id](_eval(node.args[0]))
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")

    tree = ast.parse(expression.strip(), mode="eval")
    result = _eval(tree)
    if isinstance(result, float):
        result = round(result, 10)
    return result


def calculator_tool(expression: str) -> str:
    try:
        return str(safe_eval(expression))
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return (f"Error: could not evaluate {expression!r} ({exc}). "
                "Send a pure arithmetic expression, e.g. 2*(3+4)**2 or sqrt(144).")


# ---------------------------------------------------------------------------
# Tool 2 — Wikipedia via the public REST API (stdlib only, no dependency)
# ---------------------------------------------------------------------------
def wikipedia_tool(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        return "Error: empty topic."
    title = urllib.parse.quote(topic.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(url, headers={"User-Agent": "RecentAgentDemo/1.0 (testing demo)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return f"No Wikipedia article found for '{topic}'. Try another spelling."
        return f"Wikipedia unavailable (HTTP {exc.code})."
    except Exception as exc:
        return f"Wikipedia unavailable ({type(exc).__name__})."
    extract = (data.get("extract") or "").strip()
    if not extract:
        return f"No summary text available for '{topic}'."
    sentences = re.split(r"(?<=[.!?])\s+", extract)
    return " ".join(sentences[:4])


# ---------------------------------------------------------------------------
# Tool 3 — current date & time
# ---------------------------------------------------------------------------
def datetime_tool(_: str = "") -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%A, %d %B %Y — %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Tool 4 — knowledge_base_search: TF-IDF from scratch over the sample docs
# ---------------------------------------------------------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "not", "no", "so", "such",
    "can", "could", "will", "would", "should", "may", "might", "must", "do",
    "does", "did", "have", "has", "had", "you", "your", "we", "our", "they",
    "their", "he", "she", "his", "her", "i", "me", "my", "than", "then", "when",
    "what", "which", "who", "whom", "how", "why", "where", "also", "into",
    "more", "most", "other", "some", "any", "each", "few", "all", "own",
}
TOKEN_RE = re.compile(r"[a-z0-9']+")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def _tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def _chunk(text: str) -> list[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        chunks.append(text[start:start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 40]


def _build_index() -> list[dict]:
    """Chunk sample_docs and build a tiny TF-IDF index (pure Python)."""
    docs_dir = BASE_DIR / "sample_docs"
    chunks: list[dict] = []
    if docs_dir.is_dir():
        for path in sorted(docs_dir.glob("*.txt")):
            raw = path.read_text(encoding="utf-8", errors="replace")
            for i, piece in enumerate(_chunk(raw)):
                chunks.append({"source": path.name, "chunk_id": i, "text": piece,
                               "tf": _term_freq(_tokenize(piece))})
    n = len(chunks) or 1
    df: dict[str, int] = {}
    for ch in chunks:
        for term in ch["tf"]:
            df[term] = df.get(term, 0) + 1
    idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
    for ch in chunks:
        ch["vec"] = {t: f * idf.get(t, 0.0) for t, f in ch["tf"].items()}
    return chunks


def _term_freq(tokens: list[str]) -> dict[str, float]:
    freq: dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0.0) + 1.0
    total = len(tokens) or 1
    return {t: f / total for t, f in freq.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0


KB_CHUNKS = _build_index()


def make_kb_search(hits_sink: list):
    """Build the knowledge_base_search tool bound to a per-request hits sink
    (same factory pattern as the local app's make_kb_tool)."""
    def _run(query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Error: empty knowledge base query."
        if not KB_CHUNKS:
            return "Knowledge base is empty in this demo."
        qvec = _term_freq(_tokenize(query))
        scored = sorted(
            ((_cosine(qvec, ch["vec"]), ch) for ch in KB_CHUNKS),
            key=lambda pair: pair[0], reverse=True,
        )[:TOP_K]
        hits_sink.clear()
        hits_sink.extend(
            {"source": ch["source"], "chunk_id": ch["chunk_id"], "score": round(score, 3), "text": ch["text"]}
            for score, ch in scored if score > 0.02
        )
        if not hits_sink:
            return ("No relevant passages found in the knowledge base. "
                    "It may not contain the answer — answer from your own "
                    "knowledge or use another tool.")
        blocks = [f"[{i + 1}] (source: {h['source']}, relevance: {h['score']:.2f})\n{h['text']}"
                  for i, h in enumerate(hits_sink)]
        return "\n\n".join(blocks)
    return _run


# ---------------------------------------------------------------------------
# ReAct agent loop (ported from core/agent.py — same prompts & parsing)
# ---------------------------------------------------------------------------
_FINAL_RE = re.compile(r"Final Answer\s*:(.*)", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z_][\w\-]*)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)


def _clean_thought(fragment: str) -> str:
    fragment = re.sub(r"^\s*Thought\s*:\s*", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"Observation:.*$", "", fragment, flags=re.DOTALL | re.IGNORECASE)
    return fragment.strip()


def _clean_action_input(raw: str) -> str:
    for stop in ("Observation:", "Thought:"):
        pos = raw.lower().find(stop.lower())
        if pos != -1:
            raw = raw[:pos]
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```") and len(raw) >= 6:
        raw = raw[3:-3].strip()
        lines = raw.split("\n", 1)
        if len(lines) == 2 and lines[0].strip() and " " not in lines[0].strip() \
                and len(lines[0].strip()) < 15:
            raw = lines[1].strip()
    return raw.strip().strip('"').strip()


def parse_react(text: str) -> dict:
    final = _FINAL_RE.search(text)
    if final:
        return {"type": "final", "thought": _clean_thought(text[: final.start()]),
                "answer": final.group(1).strip()}
    action = _ACTION_RE.search(text)
    if action:
        raw_input = _INPUT_RE.search(text)
        raw = raw_input.group(1) if raw_input else ""
        return {"type": "action", "name": action.group(1).strip().lower(),
                "input": _clean_action_input(raw),
                "thought": _clean_thought(text[: action.start()])}
    return {"type": "final", "thought": "", "answer": text.strip()}


def _dispatch(tools: dict, name: str, tool_input: str) -> str:
    tool = tools.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(tools)}."
    try:
        observation = tool(tool_input)
    except Exception as exc:  # tools never crash the loop
        observation = f"Tool error ({type(exc).__name__}): {exc}"
    observation = str(observation).strip() or "(empty observation)"
    if len(observation) > TOOL_OBS_CAP:
        observation = observation[:TOOL_OBS_CAP] + " …[truncated]"
    return observation

SYSTEM_TEMPLATE = """You are RecentAgent, a precise and helpful AI assistant. You answer questions by reasoning step by step and calling tools when you need information or computation.

# Available tools
{tool_lines}

# Protocol — follow EXACTLY
When you need a tool, reply with ONLY:
Thought: <one short sentence: what you know so far and what you need next>
Action: <tool name, exactly one of: {tool_names}>
Action Input: <the single input string for that tool>

The system executes the tool and replies with:
Observation: <tool result>

You may repeat Thought/Action/Action Input up to {max_iterations} times.

# Rules
1. For ANY question that could be answered from the knowledge base documents, call `knowledge_base_search` BEFORE answering from memory.
2. One tool call per message. Never write "Observation:" yourself. Never invent tool results.
3. If a tool errors or returns nothing useful, change your approach or answer from your own knowledge (saying so).
4. When you have enough information, reply with ONLY:
Thought: I now know the final answer
Final Answer: <a complete, well-structured answer. If you used knowledge_base_search, cite passages inline like [source: document_name]>
5. For simple greetings or small talk, skip tools and answer directly with Final Answer.
"""

TOOL_NAMES = ["knowledge_base_search", "calculator", "wikipedia", "current_datetime"]

SYSTEM_PROMPT = SYSTEM_TEMPLATE.format(
    tool_lines="\n".join(
        "- knowledge_base_search: Search the demo knowledge base (RAG over the sample documents: AI agents, RAG, AI governance, RecentAgent user guide). Use this FIRST for questions about these topics. (Action Input: <search query>)\n"
        "- calculator: Evaluate arithmetic expressions exactly. Supports + - * / // % **, parentheses, sqrt/log/sin/cos, pi and e. (Action Input: <math expression>)\n"
        "- wikipedia: Fetch a short encyclopedia summary of a well-known topic, person, place or concept. (Action Input: <topic>)\n"
        "- current_datetime: Get today's date and current time (UTC). (Action Input: empty string)"
    ),
    tool_names=", ".join(TOOL_NAMES),
    max_iterations=MAX_ITERATIONS,
)


def run_agent(question: str, history: list | None = None) -> dict:
    """Run the ReAct loop in STATELESS mode.

    Each round is a fresh [system, single-user-turn] request that carries the
    question plus all observations so far. The free-tier Gemini endpoint
    intermittently returns empty completions for multi-turn histories
    (thinking-model signature round-trip), so the demo rebuilds one user turn
    per round instead — same protocol, same trace, fully stateless.
    """
    history = history or []
    hits: list[dict] = []  # per-request sources sink (thread-safe: one list per run)
    tools = {
        "knowledge_base_search": make_kb_search(hits),
        "calculator": calculator_tool,
        "wikipedia": wikipedia_tool,
        "current_datetime": datetime_tool,
    }

    history_text = ""
    for role, content in history[-4:]:
        if role in ("user", "assistant") and content:
            history_text += ("User: " if role == "user" else "Assistant: ") + str(content)[:800] + "\n\n"

    def build_messages(observations: list[str], extra_instruction: str = "") -> list[dict]:
        parts = []
        if history_text:
            parts.append("Earlier conversation (for context):\n" + history_text.strip())
        parts.append("Question: " + question)
        for i, obs in enumerate(observations, 1):
            parts.append(f"Round {i} result — Observation: {obs}")
        if observations:
            parts.append("Continue now: reply with your next Thought/Action/Action Input, or your Final Answer.")
        if extra_instruction:
            parts.append(extra_instruction)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    steps: list[dict] = []
    observations: list[str] = []
    answer: str | None = None
    rounds = 0
    t0 = time.monotonic()
    deadline = t0 + WALL_BUDGET_S

    for iteration in range(1, MAX_ITERATIONS + 1):
        text, _assistant_msg = llm_chat(build_messages(observations), deadline=deadline)
        parsed = parse_react(text)

        if parsed["type"] == "final":
            answer = parsed["answer"]
            steps.append({"iteration": iteration, "thought": parsed["thought"], "final": answer})
            break

        rounds = iteration
        observation = _dispatch(tools, parsed["name"], parsed["input"])
        observations.append(observation)
        steps.append({"iteration": iteration, "thought": parsed["thought"],
                      "action": parsed["name"], "action_input": parsed["input"],
                      "observation": observation})

        if time.monotonic() - t0 > WALL_BUDGET_S:
            break

    if answer is None:  # budget exhausted -> force closure
        messages = build_messages(
            observations,
            "You have reached the tool-round limit. Reply now with your best "
            "answer in the format:\nFinal Answer: <answer>",
        )
        text = llm_chat(messages, max_tokens=900, timeout=15.0, attempts=1)[0]
        parsed = parse_react(text)
        answer = parsed["answer"] if parsed["type"] == "final" else text.strip()
        steps.append({"iteration": MAX_ITERATIONS,
                      "thought": "(tool limit reached — forced final answer)",
                      "final": answer})

    return {"answer": answer, "steps": steps, "sources": list(hits),
            "tool_rounds": rounds}


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (BASE_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/logo.png")
async def logo() -> FileResponse:
    return FileResponse(BASE_DIR / "logo.png", media_type="image/png")


@app.get("/favicon.png")
async def favicon() -> FileResponse:
    return FileResponse(BASE_DIR / "favicon.png", media_type="image/png")


@app.get("/favicon-32.png")
async def favicon_32() -> FileResponse:
    return FileResponse(BASE_DIR / "favicon-32.png", media_type="image/png")


@app.get("/apple-touch-icon.png")
async def apple_touch_icon() -> FileResponse:
    return FileResponse(BASE_DIR / "apple-touch-icon.png", media_type="image/png")


@app.get("/api/kb")
async def kb() -> dict:
    """Knowledge-base panel data for the demo UI (mirrors the local app's
    sidebar KB section: document list, chunk counts, stats)."""
    docs: dict[str, dict] = {}
    for ch in KB_CHUNKS:  # real chunk counts from the TF-IDF index
        entry = docs.setdefault(ch["source"], {"name": ch["source"], "chunks": 0})
        entry["chunks"] += 1
    for path in sorted((BASE_DIR / "sample_docs").glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        entry = docs.setdefault(path.name, {"name": path.name, "chunks": 0})
        entry["size"] = len(raw)
        entry["words"] = len(raw.split())
        entry["preview"] = " ".join(raw.strip().split())[:180]
    return {
        "demo": True,
        "documents": sorted(docs.values(), key=lambda d: d["name"]),
        "total_chunks": len(KB_CHUNKS),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedder": "TF-IDF (from scratch)",
    }


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "demo": True,
        "model": MODEL,
        "llm_configured": bool(os.environ.get("GEMINI_API") or os.environ.get("GOOGLE_API_KEY")),
        "chunks": len(KB_CHUNKS),
        "documents": sorted({c["source"] for c in KB_CHUNKS}),
    }


@app.get("/api/llm-debug")
async def llm_debug(model: str | None = None, calls: int = 4):
    """Light LLM diagnostic from this server's network position. Optional
    ?model=X&calls=N runs N full-prompt calls against a specific model to
    compare reliability (no secrets involved)."""
    import urllib.parse as _up
    model = model or MODEL
    calls = max(1, min(6, calls))
    prod = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Question: According to my documents, what are the stages of a RAG pipeline?"}]
    out = []
    for i in range(calls):
        t0 = time.monotonic()
        try:
            text, _m = await run_in_threadpool(llm_chat, [dict(m) for m in prod], 1800, 20.0, 1, None, model)
            out.append({"i": i + 1, "ok": True, "s": round(time.monotonic() - t0, 2), "head": text[:60]})
        except LLMError as exc:
            out.append({"i": i + 1, "ok": False, "s": round(time.monotonic() - t0, 2), "err": str(exc)[:150]})
    ok = sum(1 for r in out if r["ok"])
    return {"model": model, "ok": ok, "of": calls, "results": out}


@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body."})
    question = str(body.get("question") or "").strip()
    history = body.get("history") or []
    if not question:
        return JSONResponse(status_code=400, content={"error": "question is required"})
    if len(question) > 2000:
        return JSONResponse(status_code=400, content={"error": "question too long (max 2000 chars)"})
    if not isinstance(history, list):
        history = []

    try:
        result = await run_in_threadpool(run_agent, question, history)
    except LLMError as exc:
        return JSONResponse(status_code=502, content={"error": f"LLM error: {exc}"})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Agent error: {type(exc).__name__}: {exc}"})

    return {
        "answer": result["answer"],
        "steps": result["steps"],
        "sources": result["sources"],
        "tool_rounds": result["tool_rounds"],
    }
