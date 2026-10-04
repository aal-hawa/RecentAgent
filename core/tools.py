"""Tool registry for the agent.

Each tool is a small, self-contained function wrapped in a `Tool` record that
carries the name/description the LLM sees in its prompt. Tools always return
a *string* (LLMs consume text) and never raise — failures are reported back
to the model as an observation so the agent can recover and try something
else, exactly like a human would.
"""
from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    usage: str          # shown in the prompt as the expected Action Input
    func: Callable[[str], str]


# ---------------------------------------------------------------------------
# 1. Calculator — AST-based safe arithmetic evaluator
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
    """Evaluate a pure-arithmetic expression WITHOUT `eval()`.

    We parse with Python's `ast` module and only whitelisted node types are
    executed. Anything else (attribute access, names, calls to unknown
    functions, strings...) is rejected. This blocks classic code-injection
    payloads such as `__import__('os').system('rm -rf /')` while still
    supporting real math.
    """

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
# 2. Web search — DuckDuckGo via `ddgs`, no API key required
# ---------------------------------------------------------------------------
def web_search_tool(query: str, max_results: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "Error: empty search query."
    try:
        try:                                   # new package name (2025+)
            from ddgs import DDGS
        except ImportError:                    # legacy name fallback
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No web results found for '{query}'."
        blocks = [
            f"[{i + 1}] {r.get('title', '')}\nURL: {r.get('href', '')}\n{r.get('body', '')}"
            for i, r in enumerate(results)
        ]
        return "\n\n".join(blocks)
    except Exception as exc:  # offline / rate-limited -> agent degrades gracefully
        return (f"Web search unavailable ({type(exc).__name__}: {exc}). "
                "If the answer can be given from your own knowledge, do so and "
                "mention that live search was unavailable.")


# ---------------------------------------------------------------------------
# 3. Wikipedia lookup — good for general knowledge grounding
# ---------------------------------------------------------------------------
def wikipedia_tool(topic: str) -> str:
    topic = (topic or "").strip()
    if not topic:
        return "Error: empty topic."
    try:
        import wikipedia
    except Exception as exc:
        return f"Wikipedia library unavailable ({exc})."
    try:
        return wikipedia.summary(topic, sentences=4, auto_suggest=True)
    except wikipedia.DisambiguationError as exc:
        options = ", ".join(str(o) for o in exc.options[:8])
        return (f"'{topic}' is ambiguous. Did you mean one of: {options}? "
                "Retry with a more specific topic.")
    except wikipedia.PageError:
        return f"No Wikipedia article found for '{topic}'. Try another spelling."
    except Exception as exc:
        return f"Wikipedia unavailable ({type(exc).__name__}: {exc})."


# ---------------------------------------------------------------------------
# 4. Current date & time — trivial but grounds "what day is it" style questions
# ---------------------------------------------------------------------------
def datetime_tool(_: str = "") -> str:
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y — %H:%M:%S (%Z, UTC%z)")


# ---------------------------------------------------------------------------
# 5. Knowledge base search — the RAG tool (built as a factory so the agent
#    can collect the exact hits used, for the citations panel in the UI)
# ---------------------------------------------------------------------------
def make_kb_tool(kb, hits_sink: list, top_k: int | None = None) -> Tool:
    def _run(query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Error: empty knowledge base query."
        hits = kb.search(query, k=top_k or kb.top_k)
        hits_sink.clear()
        hits_sink.extend(hits)
        if not hits:
            return ("No relevant passages found in the knowledge base. "
                    "It may not contain the answer — answer from your own "
                    "knowledge or use another tool.")
        blocks = [
            f"[{i + 1}] (source: {h['source']}, relevance: {h['score']:.2f})\n{h['text']}"
            for i, h in enumerate(hits)
        ]
        return "\n\n".join(blocks)

    return Tool(
        name="knowledge_base_search",
        description=("Search the user's uploaded documents (RAG). Use this FIRST "
                     "for any question that could be answered from the indexed "
                     "documents: internal guides, notes, PDFs, policies."),
        usage="<search query>",
        func=_run,
    )


# ---------------------------------------------------------------------------
def build_default_tools(kb=None, include_net: bool = True,
                        top_k: int | None = None) -> tuple[dict[str, Tool], list]:
    """Assemble the standard tool set. Returns (tools_by_name, hits_sink).

    include_net=False yields a 100% offline agent (no web_search / wikipedia).
    top_k overrides how many knowledge-base passages are retrieved per query.
    """
    hits_sink: list = []
    tools: list[Tool] = []
    if kb is not None:
        tools.append(make_kb_tool(kb, hits_sink, top_k=top_k))
    tools.extend([
        Tool(name="calculator",
             description="Evaluate arithmetic expressions exactly. Supports + - * / "
                         "// % **, parentheses, sqrt/log/sin/cos, pi and e.",
             usage="<math expression>", func=calculator_tool),
    ])
    if include_net:
        tools.extend([
            Tool(name="web_search",
                 description="Search the public web (DuckDuckGo) for current events, "
                             "facts or anything not in the knowledge base.",
                 usage="<search query>", func=web_search_tool),
            Tool(name="wikipedia",
                 description="Fetch a short encyclopedia summary of a well-known "
                             "topic, person, place or concept.",
                 usage="<topic>", func=wikipedia_tool),
        ])
    tools.append(
        Tool(name="current_datetime",
             description="Get today's date and current local time.",
             usage="empty string", func=datetime_tool),
    )
    return {t.name: t for t in tools}, hits_sink
