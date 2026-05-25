"""
deployment/hf_spaces/app.py
Qwen2.5-0.5B-Instruct on Hugging Face Spaces (CPU, free tier).
Drop this file + requirements.txt into a new Gradio Space and push.
"""

import os
import time
import json
import uuid
import gradio as gr
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MEMORY_WINDOW = 10
MAX_NEW_TOKENS = 512
LOG_FILE = "turns.jsonl"

# ── Model load ────────────────────────────────────────────────────────────────

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.getenv("HF_TOKEN"))
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto",
    token=os.getenv("HF_TOKEN"),
)
print("Ready.")

# ── Guardrails ────────────────────────────────────────────────────────────────

_INPUT_PATTERNS = [
    r"ignore (all |previous |prior )?instructions",
    r"you are now (an? )?",
    r"act as (if you have no|an? AI without)",
    r"jailbreak|DAN mode|developer mode",
    r"how (to|do I) (make|build|synthesize) .{0,30}(bomb|weapon|drug|poison)",
]

_OUTPUT_PATTERNS = [
    r"(here (is|are)|sure).{0,20}(step|instruction).{0,30}(make|build|create).{0,20}(bomb|weapon|drug)",
]

REFUSAL = "I'm sorry, I can't help with that. Feel free to ask me something else."
REDACTED = "⚠️ Response withheld — unsafe content detected."


def _matches(text: str, patterns: list) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


# ── Inference ─────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Answer clearly and concisely. Refuse harmful requests."
)


def _generate(messages: list) -> tuple[str, int, int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][n_in:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, n_in, len(new_tokens)


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(session_id, turn, latency_ms, in_tok, out_tok, blocked):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": session_id,
        "turn": turn,
        "model": MODEL_ID,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "blocked": blocked,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Chat handler ──────────────────────────────────────────────────────────────

_SESSION = str(uuid.uuid4())[:8]
_turn = 0


def chat(user_msg: str, history: list):
    global _turn
    _turn += 1

    if _matches(user_msg, _INPUT_PATTERNS):
        _log(_SESSION, _turn, 0, 0, 0, blocked=True)
        history.append((user_msg, REFUSAL))
        return "", history

    # Build messages with sliding window
    messages = [{"role": "system", "content": SYSTEM}]
    for u, a in history[-(MEMORY_WINDOW):]:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": user_msg})

    t0 = time.perf_counter()
    response, in_tok, out_tok = _generate(messages)
    latency_ms = (time.perf_counter() - t0) * 1000

    if _matches(response, _OUTPUT_PATTERNS):
        response = REDACTED

    _log(_SESSION, _turn, latency_ms, in_tok, out_tok, blocked=False)
    history.append((user_msg, response))
    return "", history


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="OSS Assistant — Qwen2.5-0.5B") as demo:
    gr.Markdown(
        "## Qwen2.5-0.5B-Instruct\n"
        "Open-source assistant with sliding-window memory and safety guardrails.\n\n"
        "> Running on CPU — expect 4–8s per response. Cold start ~30s."
    )
    chatbot = gr.Chatbot(height=400)
    state = gr.State([])
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything...", show_label=False, scale=8)
        gr.Button("Send", variant="primary", scale=1).click(
            chat, [msg, state], [msg, chatbot]
        )
        gr.Button("Clear", scale=1).click(lambda: ([], []), [], [state, chatbot])
    msg.submit(chat, [msg, state], [msg, chatbot])
    gr.Examples(
        ["What is the capital of Australia?", "Explain quantum entanglement simply.",
         "What's 17 × 23?", "Write a haiku about machine learning."],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.launch()