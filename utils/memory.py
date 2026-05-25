from dataclasses import dataclass

@dataclass
class Message:
    role: str   # "user" | "assistant"
    content: str

class ConversationMemory:
    """Sliding-window memory. Keeps last `max_turns` user+assistant pairs."""

    def __init__(self, system_prompt: str = "", max_turns: int = 10):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._history: list[Message] = []

    def add_user(self, content: str):
        self._history.append(Message("user", content))
        self._evict()

    def add_assistant(self, content: str):
        self._history.append(Message("assistant", content))

    def to_messages(self) -> list[dict]:
        """OpenAI-style: system message first, then history."""
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs += [{"role": m.role, "content": m.content} for m in self._history]
        return msgs

    def to_anthropic_messages(self) -> tuple[str, list[dict]]:
        """Anthropic SDK keeps system prompt separate."""
        msgs = [{"role": m.role, "content": m.content} for m in self._history]
        return self.system_prompt, msgs

    def clear(self):
        self._history.clear()

    def __len__(self):
        return len(self._history)

    def _evict(self):
        pairs = 0
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].role == "user":
                pairs += 1
            if pairs > self.max_turns:
                self._history = self._history[i + 1:]
                # Drop orphaned leading assistant message
                while self._history and self._history[0].role == "assistant":
                    self._history.pop(0)
                return