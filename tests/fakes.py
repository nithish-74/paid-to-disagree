"""CPU stand-in for a model backend, shared by tests."""

from __future__ import annotations

from typing import Callable

from data.pressure import Message


class FakeBackend:
    """`reply(conversation) -> str`; `probs(conversation) -> dict` for option-token probabilities."""

    def __init__(
        self,
        reply: Callable[[list[Message]], str],
        probs: Callable[[list[Message]], dict[str, float]] | None = None,
    ) -> None:
        self.reply = reply
        self.probs = probs
        self.chat_calls: list[list[list[Message]]] = []
        self.prob_calls = 0

    def chat(self, conversations, temperature, max_tokens, seed=None):
        self.chat_calls.append(conversations)
        return [self.reply(c) for c in conversations]

    def next_token_probs(self, conversations, assistant_prefix, candidates):
        self.prob_calls += 1
        if self.probs is None:
            return [{c: 0.0 for c in candidates} for _ in conversations]
        return [{c: self.probs(conv).get(c, 0.0) for c in candidates} for conv in conversations]


def last_user(conv: list[Message]) -> str:
    return [m for m in conv if m["role"] == "user"][-1]["content"]


def answer_json(answer: str, conf: float = 0.8, pp: dict | None = None, reason: str = "because") -> str:
    import json

    obj = {"reason": reason, "answer": answer, "confidence": conf}
    if pp is not None:
        obj["peer_prediction"] = pp
    return json.dumps(obj)
