SYSTEM_PROMPT = """You are a helpful, honest personal assistant.
- Be concise and clear.
- If you don't know something, say so — don't make things up.
- Decline harmful, illegal, or unethical requests.
"""

FRONTIER_MODEL  = "claude-sonnet-4-20250514"
OSS_MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
HF_API_URL      = f"https://api-inference.huggingface.co/models/{OSS_MODEL_ID}"

MAX_TURNS       = 10
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.7