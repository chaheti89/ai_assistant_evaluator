"""
Qwen2.5-0.5B-Instruct on Hugging Face Spaces (CPU, free tier).
Includes: sliding-window memory, guardrails, observability, tool use.
"""

import os, re, time, json, uuid, datetime, math, logging
import gradio as gr
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
MEMORY_WINDOW  = 10
MAX_NEW_TOKENS = 512
LOG_FILE = "turns.jsonl"

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, device_map="auto",
    token=os.getenv("HF_TOKEN"),
)
print("Ready.")

# ── Tools ─────────────────────────────────────────────────────────────────────
def _calculator(expr: str) -> str:
    try:
        result = eval(expr.replace("^","**"), {"__builtins__": {}}, {
            "sqrt": math.sqrt, "log": math.log, "pi": math.pi,
            "e": math.e, "abs": abs, "round": round,
        })
        return str(round(result, 6)) if isinstance(result, float) else str(result)
    except Exception as ex:
        return f"Error: {ex}"

def _get_datetime(_: str = "") -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _web_search(query: str) -> str:
    try:
        r = requests.get("https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            timeout=5)
        d = r.json()
        ans = d.get("AbstractText") or d.get("Answer") or \
              (d.get("RelatedTopics") or [{}])[0].get("Text", "")
        return ans[:400] if ans else "No result found."
    except Exception as ex:
        return f"Search error: {ex}"

TOOLS = {
    "calculator":   {"fn": _calculator,   "desc": "Evaluates a math expression."},
    "get_datetime": {"fn": _get_datetime, "desc": "Returns current UTC date and time."},
    "web_search":   {"fn": _web_search,   "desc": "Fetches a short factual web answer."},
}
_TOOL_LIST = "\n".join(f"- {k}: {v['desc']}" for k,v in TOOLS.items())
_TOOL_RE = re.compile(r"TOOL:\s*(\w+)\s*\nINPUT:\s*(.*)", re.I | re.DOTALL)

# ── Guardrails ────────────────────────────────────────────────────────────────
_IN_PATTERNS = [
    r"ignore (all |previous |prior )?instructions",
    r"you are now (an? )?", r"jailbreak|DAN mode",
    r"how (to|do I) (make|build|synthesize) .{0,30}(bomb|weapon|drug)",
]
_OUT_PATTERNS = [
    r"(here (is|are)|sure).{0,20}step.{0,30}(make|build).{0,20}(bomb|weapon|drug)",
]
REFUSAL = "I'm sorry, I can't help with that. Feel free to ask something else."
REDACTED = "⚠️ Response withheld — unsafe content detected."

def _blocked(text, patterns):
    return any(re.search(p, text, re.I) for p in patterns)

# ── Inference ─────────────────────────────────────────────────────────────────
_SYSTEM = f"""You are a helpful assistant with access to tools.

Available tools:
{_TOOL_LIST}

To use a tool respond ONLY with:
TOOL: <tool_name>
INPUT: <input>

Otherwise answer directly."""

def _generate(messages: list) -> tuple[str, int, int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tokenizer.eos_token_id)
    new_tokens = out[0][n_in:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), n_in, len(new_tokens)

def _run_tools(first_response: str, messages: list) -> tuple[str, list[dict]]:
    """If model called a tool, execute it and get final answer."""
    tool_log = []
    response = first_response
    for _ in range(2):  # max 2 tool calls
        match = _TOOL_RE.search(response.strip())
        if not match:
            break
        tool_name = match.group(1).strip().lower()
        tool_input = match.group(2).strip()
        if tool_name not in TOOLS:
            break
        result = TOOLS[tool_name]["fn"](tool_input)
        tool_log.append({"tool": tool_name, "input": tool_input, "result": result})
        logger.info("Tool: %s(%r) → %s", tool_name, tool_input, result[:80])
        # Feed result back
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content":
            f"Tool result for {tool_name}({tool_input}):\n{result}\n\nNow answer the original question."})
        response, _, _ = _generate(messages)
    return response, tool_log

# ── Logging ───────────────────────────────────────────────────────────────────
SESSION = str(uuid.uuid4())[:8]
_turn = 0

def _log(latency_ms, in_tok, out_tok, blocked, tools_used):
    global _turn
    _turn += 1
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "session": SESSION, "turn": _turn, "model": MODEL_ID,
        "latency_ms": round(latency_ms, 1), "input_tokens": in_tok,
        "output_tokens": out_tok, "blocked": blocked,
        "tools_used": [t["tool"] for t in tools_used],
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Chat handler ──────────────────────────────────────────────────────────────
def chat(user_msg: str, history: list, show_tools: bool):
    if _blocked(user_msg, _IN_PATTERNS):
        _log(0, 0, 0, True, [])
        history.append((user_msg, REFUSAL))
        return "", history, ""

    # Build messages with sliding window + tool system prompt
    messages = [{"role": "system", "content": _SYSTEM}]
    for u, a in history[-(MEMORY_WINDOW):]:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user_msg})

    t0 = time.perf_counter()
    response, in_tok, out_tok = _generate(messages)

    # Tool use
    response, tool_log = _run_tools(response, messages)
    latency_ms = (time.perf_counter() - t0) * 1000

    if _blocked(response, _OUT_PATTERNS):
        response = REDACTED

    _log(latency_ms, in_tok, out_tok, False, tool_log)

    tool_info = ""
    if tool_log and show_tools:
        tool_info = "\n".join(
            f"🔧 `{t['tool']}({t['input']})` → {t['result']}" for t in tool_log
        )

    history.append((user_msg, response))
    return "", history, tool_info

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="OSS Assistant — Qwen2.5-0.5B") as demo:
    gr.Markdown(
        "## Qwen2.5-0.5B-Instruct\n"
        "Open-source assistant with **tool use**, sliding-window memory, and safety guardrails.\n\n"
        "> Running on CPU — expect 4–8s per response."
    )
    chatbot = gr.Chatbot(height=420)
    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything...", show_label=False, scale=8)
        send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)

    show_tools = gr.Checkbox(label="Show tool calls", value=True)
    tool_output = gr.Markdown(label="Tool trace")

    gr.Examples([
        ["What is 1234 * 5678?"],
        ["What is the current UTC time?"],
        ["Who invented the telephone?"],
        ["What's sqrt(2) + pi?"],
        ["Write a haiku about machine learning."],
    ], inputs=msg)

    send.click(chat, [msg, state, show_tools], [msg, chatbot, tool_output])
    msg.submit(chat, [msg, state, show_tools], [msg, chatbot, tool_output])
    clear.click(lambda: ([], [], ""), [], [state, chatbot, tool_output])

if __name__ == "__main__":
    demo.launch()