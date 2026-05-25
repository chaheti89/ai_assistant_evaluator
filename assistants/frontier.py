import os
import anthropic
from assistants.base import BaseAssistant
from utils.memory import ConversationMemory
from utils.config import FRONTIER_MODEL, MAX_NEW_TOKENS, TEMPERATURE

class FrontierAssistant(BaseAssistant):
    def __init__(self):
        super().__init__()
        self.name = "Claude Sonnet (Frontier)"
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _generate(self, memory: ConversationMemory) -> str:
        system_prompt, messages = memory.to_anthropic_messages()
        response = self.client.messages.create(
            model=FRONTIER_MODEL,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text