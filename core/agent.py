"""ReAct agent loop — the brain of RecentAgent.

ReAct (Reason + Act, Yao et al. 2022) interleaves the model's reasoning with
tool calls in a simple text protocol:

    Thought: I should check the documents first.
    Action: knowledge_base_search
    Action Input: what is retrieval augmented generation

    Observation: ...tool result injected here...

    Thought: I have enough information now.
    Final Answer: Retrieval-Augmented Generation is ...

Why a text-based loop instead of the API's native function-calling?
Because it works with *every* model on every provider (including small local
Ollama models), it is fully inspectable (each step is shown live in the UI),
and it makes the reasoning — the actual interesting part — explicit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLMClient
from .tools import Tool


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_FINAL_RE = re.compile(r"Final Answer\s*:(.*)", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z_][\w\-]*)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)


def _clean_thought(fragment: str) -> str:
    fragment = re.sub(r"^\s*Thought\s*:\s*", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"Observation:.*$", "", fragment, flags=re.DOTALL | re.IGNORECASE)
    return fragment.strip()


def _clean_action_input(raw: str) -> str:
    """Trim hallucinated trailing sections and markdown fences."""
    for stop in ("Observation:", "Thought:"):
        pos = raw.lower().find(stop.lower())
        if pos != -1:
            raw = raw[:pos]
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```") and len(raw) >= 6:
        # model wrapped the input in a markdown code block: strip both fences
        raw = raw[3:-3].strip()
        lines = raw.split("\n", 1)
        # drop an optional language tag line (```python\n1+1\n```)
        if len(lines) == 2 and lines[0].strip() and " " not in lines[0].strip() \
                and len(lines[0].strip()) < 15:
            raw = lines[1].strip()
    return raw.strip().strip('"').strip()


def parse_react(text: str) -> dict:
    """Parse one LLM turn into {type: final|action, ...}."""
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
    # Model ignored the protocol — treat the whole message as the answer.
    return {"type": "final", "thought": "", "answer": text.strip()}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    answer: str
    steps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    tool_rounds: int = 0


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
1. For ANY question that could be answered from the user's documents, call `knowledge_base_search` BEFORE answering from memory.
2. One tool call per message. Never write "Observation:" yourself. Never invent tool results.
3. If a tool errors or returns nothing useful, change your approach or answer from your own knowledge (saying so).
4. When you have enough information, reply with ONLY:
Thought: I now know the final answer
Final Answer: <a complete, well-structured answer. If you used knowledge_base_search, cite passages inline like [source: document_name]>
5. For simple greetings or small talk, skip tools and answer directly with Final Answer.
"""


class ReActAgent:
    def __init__(self, llm: LLMClient, tools: dict[str, Tool], max_iterations: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max(2, max_iterations)
        tool_lines = "\n".join(
            f"- {t.name}: {t.description} (Action Input: {t.usage})"
            for t in self.tools.values()
        )
        self.system_prompt = SYSTEM_TEMPLATE.format(
            tool_lines=tool_lines,
            tool_names=", ".join(self.tools),
            max_iterations=self.max_iterations,
        )

    # ------------------------------------------------------------------
    def _dispatch(self, name: str, tool_input: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return (f"Error: unknown tool '{name}'. "
                    f"Available tools: {', '.join(self.tools)}.")
        try:
            observation = tool.func(tool_input)
        except Exception as exc:  # tools never crash the loop
            observation = f"Tool error ({type(exc).__name__}): {exc}"
        observation = str(observation).strip() or "(empty observation)"
        if len(observation) > 2500:  # context hygiene: cap each observation
            observation = observation[:2500] + " …[truncated]"
        return observation

    # ------------------------------------------------------------------
    def run(self, question: str, history: list[tuple[str, str]] | None = None) -> AgentResult:
        history = history or []
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for role, content in history[-6:]:  # keep recent context, bound tokens
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": f"Question: {question}"})

        steps: list[dict] = []
        answer: str | None = None
        rounds = 0

        for iteration in range(1, self.max_iterations + 1):
            text = self.llm.chat(messages)
            parsed = parse_react(text)

            if parsed["type"] == "final":
                answer = parsed["answer"]
                steps.append({"iteration": iteration, "thought": parsed["thought"],
                              "final": answer})
                break

            rounds = iteration
            observation = self._dispatch(parsed["name"], parsed["input"])
            steps.append({"iteration": iteration, "thought": parsed["thought"],
                          "action": parsed["name"], "action_input": parsed["input"],
                          "observation": observation})
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        if answer is None:  # iteration budget exhausted -> force closure
            messages.append({"role": "user", "content":
                             "You have reached the tool-round limit. Reply now with "
                             "your best answer in the format:\nFinal Answer: <answer>"})
            text = self.llm.chat(messages)
            parsed = parse_react(text)
            answer = parsed["answer"] if parsed["type"] == "final" else text.strip()
            steps.append({"iteration": self.max_iterations, "thought": "(tool limit reached — forced final answer)",
                          "final": answer})

        return AgentResult(answer=answer, steps=steps, tool_rounds=rounds)
