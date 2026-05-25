"""
deployment/guardrails.py
Wraps any BaseAssistant with input + output safety filtering.

Usage:
    from deployment.guardrails import GuardrailedAssistant
    safe = GuardrailedAssistant(FrontierAssistant(), mode="block")
    response = safe.chat("ignore previous instructions")
    # → "I'm sorry, I can't help with that request."
"""

from __future__ import annotations
import re
import logging
from assistants.base import BaseAssistant

logger = logging.getLogger(__name__)

REFUSAL = "I'm sorry, I can't help with that request. If you have a different question, I'm happy to assist."
REDACTED = "⚠️ [Response withheld — potential unsafe content detected]"

_INPUT_RULES = [
    ("prompt_injection", [
        r"ignore (all |previous |prior )?instructions",
        r"disregard (all |your )?previous",
        r"you are now (an? )?",
        r"act as (if you have no|an? AI without)",
        r"pretend (you have no|you are)",
        r"jailbreak|DAN mode|developer mode",
    ]),
    ("harmful_request", [
        r"how (to|do I) (make|build|synthesize|create) .{0,30}(bomb|weapon|explosive|drug|poison|malware)",
        r"step.by.step.*(kill|harm|attack|hack)",
        r"(suicide|self.harm) method",
    ]),
    ("pii_exfiltration", [
        r"(credit card|ssn|social security).*(number|details)",
        r"give me.*(password|api key|secret key)",
    ]),
]

_OUTPUT_RULES = [
    ("unsafe_content", [
        r"(here (is|are)|sure).{0,20}step.{0,30}(make|build|create).{0,20}(bomb|weapon|drug)",
    ]),
    ("pii_leak", [
        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",   # SSN-like
        r"\b4[0-9]{12}(?:[0-9]{3})?\b",           # Visa-like
    ]),
]


def _check(text: str, rules: list) -> tuple[bool, str]:
    lower = text.lower()
    for category, patterns in rules:
        for p in patterns:
            if re.search(p, lower, re.I):
                return True, category
    return False, ""


class GuardrailedAssistant:
    def __init__(self, assistant: BaseAssistant, mode: str = "block"):
        assert mode in ("block", "warn"), "mode must be 'block' or 'warn'"
        self.assistant = assistant
        self.mode = mode
        self.blocked_inputs = 0
        self.blocked_outputs = 0

    def chat(self, user_message: str) -> str:
        triggered, category = _check(user_message, _INPUT_RULES)
        if triggered:
            self.blocked_inputs += 1
            logger.warning("[INPUT BLOCKED] category=%s", category)
            if self.mode == "block":
                return REFUSAL
            # warn mode: log but still call model

        response = self.assistant.chat(user_message)

        triggered, category = _check(response, _OUTPUT_RULES)
        if triggered:
            self.blocked_outputs += 1
            logger.warning("[OUTPUT BLOCKED] category=%s", category)
            return REDACTED

        return response

    def reset(self) -> None:
        self.assistant.reset()
        self.blocked_inputs = 0
        self.blocked_outputs = 0

    @property
    def stats(self) -> dict:
        return {"blocked_inputs": self.blocked_inputs, "blocked_outputs": self.blocked_outputs}