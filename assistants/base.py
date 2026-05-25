from abc import ABC, abstractmethod
from utils.memory import ConversationMemory
from utils.config import SYSTEM_PROMPT, MAX_TURNS

class BaseAssistant(ABC):
    """Both assistants share this interface. Only _generate() differs."""

    def __init__(self):
        self.name = "Assistant"
        self.memory = ConversationMemory(SYSTEM_PROMPT, MAX_TURNS)

    def chat(self, user_message: str) -> str:
        """Send a message, update memory, return response."""
        self.memory.add_user(user_message)
        response = self._generate(self.memory)
        self.memory.add_assistant(response)
        return response

    def reset(self):
        self.memory.clear()

    @abstractmethod
    def _generate(self, memory: ConversationMemory) -> str:
        """Call the model and return its reply."""
        ...