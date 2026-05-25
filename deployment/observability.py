"""
deployment/observability.py
Per-turn logging: latency, tokens, estimated cost, errors.

Usage:
    from deployment.observability import ObservableAssistant
    obs = ObservableAssistant(FrontierAssistant(), log_file="turns.jsonl")
    obs.chat("Hello!")
    obs.print_summary()
"""

from __future__ import annotations
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from assistants.base import BaseAssistant

logger = logging.getLogger(__name__)

# USD per 1M tokens — update as pricing changes
_COST_TABLE = {
    "claude-sonnet-4-5-20251001": {"in": 3.00, "out": 15.00},
    "claude-sonnet":              {"in": 3.00, "out": 15.00},
    "gpt-4.1":                    {"in": 2.00, "out": 8.00},
    "qwen2.5:0.5b":               {"in": 0.00, "out": 0.00},
    "Qwen/Qwen2.5-0.5B-Instruct": {"in": 0.00, "out": 0.00},
}


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    rates = _COST_TABLE.get(model, {"in": 0.0, "out": 0.0})
    return (in_tok * rates["in"] + out_tok * rates["out"]) / 1_000_000


class ObservableAssistant:
    def __init__(
        self,
        assistant: BaseAssistant,
        session_id: str | None = None,
        model_name: str | None = None,
        log_file: str | Path | None = "turns.jsonl",
    ):
        self.assistant = assistant
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.model = model_name or getattr(assistant, "model_name", "unknown")
        self.log_file = Path(log_file) if log_file else None
        self._turn = 0
        self._records: list[dict] = []

    def chat(self, user_message: str) -> str:
        self._turn += 1
        error = ""
        t0 = time.perf_counter()
        try:
            response = self.assistant.chat(user_message)
        except Exception as e:
            error = str(e)
            response = f"[Error: {error}]"
            logger.error("Turn %d error: %s", self._turn, error)

        latency_ms = (time.perf_counter() - t0) * 1000
        # ~4 chars per token — good enough for cost tracking
        in_tok = len(user_message) // 4
        out_tok = len(response) // 4

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "turn": self._turn,
            "model": self.model,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(_cost(self.model, in_tok, out_tok), 6),
            "error": error,
        }
        self._records.append(record)
        if self.log_file:
            try:
                with self.log_file.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as e:
                logger.warning("Log write failed: %s", e)

        logger.info("[t=%d | %.0fms | ~%dtok | $%.5f]",
                    self._turn, latency_ms, in_tok + out_tok, record["cost_usd"])
        return response

    def reset(self) -> None:
        self.assistant.reset()
        self._turn = 0
        self._records.clear()

    def summary(self) -> dict:
        if not self._records:
            return {}
        lats = [r["latency_ms"] for r in self._records]
        return {
            "session": self.session_id,
            "model": self.model,
            "turns": self._turn,
            "avg_latency_ms": round(sum(lats) / len(lats), 1),
            "p95_latency_ms": round(sorted(lats)[int(len(lats) * 0.95)], 1),
            "total_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in self._records),
            "total_cost_usd": round(sum(r["cost_usd"] for r in self._records), 5),
            "errors": sum(1 for r in self._records if r["error"]),
        }

    def print_summary(self) -> None:
        s = self.summary()
        if not s:
            print("No turns recorded.")
            return
        print(f"\n{'─'*48}")
        print(f"  {s['model']}  |  session {s['session']}")
        print(f"  turns: {s['turns']}  errors: {s['errors']}")
        print(f"  avg latency: {s['avg_latency_ms']}ms  p95: {s['p95_latency_ms']}ms")
        print(f"  total tokens: {s['total_tokens']}  cost: ${s['total_cost_usd']:.4f}")
        print(f"{'─'*48}\n")