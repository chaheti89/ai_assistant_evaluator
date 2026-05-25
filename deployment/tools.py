"""
Tool use layer for both assistants.

Gives the assistant access to:
  - calculator   : evaluates safe math expressions
  - get_datetime : returns current date and time
  - web_search   : fetches a DuckDuckGo snippet for a query

The ToolEnabledAssistant wraps any BaseAssistant. On each turn it:
  1. Asks the model if it needs a tool (ReAct-style single-pass)
  2. Runs the tool if requested
  3. Feeds the result back and gets the final answer

Works with both Frontier and OSS — no function-calling API needed.
"""

from __future__ import annotations
import re
import math
import json
import datetime
import logging
import requests
from assistants.base import BaseAssistant

logger = logging.getLogger(__name__)

# ── Tool implementations ──────────────────────────────────────────────────────

def _calculator(expression: str) -> str:
    """Evaluate a safe math expression. Supports +,-,*,/,**,sqrt,log,pi,e."""
    allowed = re.compile(r"^[\d\s\+\-\*\/\(\)\.\%\^]+$")
    expr = expression.strip()
    # Allow math function names too
    safe_expr = expr.replace("^", "**")
    try:
        result = eval(safe_expr, {"__builtins__": {}}, {
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "abs": abs, "round": round,
        })
        return str(round(result, 6)) if isinstance(result, float) else str(result)
    except Exception as ex:
        return f"Error: {ex}"


def _get_datetime(timezone_hint: str = "") -> str:
    """Return current UTC date and time."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


def _web_search(query: str) -> str:
    """Fetch a short answer from DuckDuckGo Instant Answer API (no API key needed)."""
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            timeout=5,
        )
        data = resp.json()
        # Try AbstractText first, then Answer, then first RelatedTopic
        answer = (
            data.get("AbstractText")
            or data.get("Answer")
            or (data.get("RelatedTopics") or [{}])[0].get("Text", "")
        )
        if answer:
            return answer[:400]
        return "No result found."
    except Exception as ex:
        return f"Search error: {ex}"


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = {
    "calculator": {
        "fn": _calculator,
        "description": "Evaluates a math expression. Input: the expression string.",
        "param": "expression",
    },
    "get_datetime": {
        "fn": _get_datetime,
        "description": "Returns the current UTC date and time. Input: empty string.",
        "param": "query",
    },
    "web_search": {
        "fn": _web_search,
        "description": "Searches the web for a short factual answer. Input: search query.",
        "param": "query",
    },
}

_TOOL_LIST = "\n".join(
    f"- {name}: {info['description']}"
    for name, info in TOOLS.items()
)

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """You are a helpful assistant with access to tools.

Available tools:
{tools}

To use a tool, respond ONLY with this exact format (no other text):
TOOL: <tool_name>
INPUT: <input>

If you don't need a tool, just answer directly.
""".format(tools=_TOOL_LIST)

_TOOL_RESULT_TEMPLATE = """Tool result for {tool_name}({input}):
{result}

Now answer the user's original question using this result."""

_TOOL_RE = re.compile(r"TOOL:\s*(\w+)\s*\nINPUT:\s*(.*)", re.IGNORECASE | re.DOTALL)


# ── Wrapper ───────────────────────────────────────────────────────────────────

class ToolEnabledAssistant:
    """
    Wraps any BaseAssistant with tool use via a ReAct-style single-pass loop.

    Usage:
        from deployment.tools import ToolEnabledAssistant
        from assistants.frontier import FrontierAssistant

        agent = ToolEnabledAssistant(FrontierAssistant())
        print(agent.chat("What is 1234 * 5678?"))
        print(agent.chat("What time is it?"))
        print(agent.chat("Who invented the telephone?"))
    """

    def __init__(self, assistant: BaseAssistant, max_tool_turns: int = 2):
        self.assistant = assistant
        self.max_tool_turns = max_tool_turns
        self._tool_calls: list[dict] = []

    def chat(self, user_message: str) -> str:
        # Inject tool instructions into the first message of this turn
        augmented = f"{_SYSTEM}\n\nUser: {user_message}"

        for _ in range(self.max_tool_turns):
            raw = self.assistant.chat(augmented)
            match = _TOOL_RE.search(raw.strip())

            if not match:
                # No tool needed — return the answer directly
                return raw

            tool_name = match.group(1).strip().lower()
            tool_input = match.group(2).strip()

            if tool_name not in TOOLS:
                logger.warning("Model requested unknown tool: %s", tool_name)
                return raw  # return as-is

            logger.info("Tool call: %s(%r)", tool_name, tool_input)
            result = TOOLS[tool_name]["fn"](tool_input)

            self._tool_calls.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result,
            })

            # Feed result back as next user message
            augmented = _TOOL_RESULT_TEMPLATE.format(
                tool_name=tool_name,
                input=tool_input,
                result=result,
            )

        # Exceeded max_tool_turns — ask for final answer
        return self.assistant.chat("Please give your final answer based on the tool results above.")

    def reset(self) -> None:
        self.assistant.reset()
        self._tool_calls.clear()

    @property
    def tool_call_log(self) -> list[dict]:
        return list(self._tool_calls)