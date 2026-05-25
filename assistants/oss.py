import requests
from assistants.base import BaseAssistant
from utils.memory import ConversationMemory
from utils.config import MAX_NEW_TOKENS, TEMPERATURE

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:0.5b"

class OSSAssistant(BaseAssistant):
    def __init__(self):
        super().__init__()
        self.name = "Qwen 2.5 (OSS)"

    def _generate(self, memory: ConversationMemory) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": memory.to_messages(),
            "stream": False,
            "options": {
                "num_predict": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
            }
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
        return response.json()["message"]["content"]