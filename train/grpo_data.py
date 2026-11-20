"""C7 step 2: training rows for TRL GRPOTrainer (conversational "prompt" format).

One row per (train question, pressure condition). Each row carries only what the reward
needs: row_id, qid, condition and pressure_target. Gold labels are never put in a row.

are_you_sure is two-turn: the first assistant turn is this member's cached neutral
answer (its majority sample), followed by "Are you sure? ...". Questions without a
usable cached answer are skipped for that condition.
"""

from __future__ import annotations

from typing import Any

from council.members import system_prompt
from data.build_questions import Record
from data.pressure import apply_condition, second_turn
from train.peer_cache import PeerCache

TRAIN_CONDITIONS = ("neutral", "user_wrong", "user_right", "are_you_sure")


def build_rows(
    records: list[Record],
    member: str,
    cache: PeerCache,
    conditions: tuple[str, ...] | list[str] = TRAIN_CONDITIONS,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec["split"] != "train":
            raise ValueError(f"{rec['qid']} is in split {rec['split']!r}; only train questions may be used")
        for cond in conditions:
            p = apply_condition(rec, cond, seed)
            if p.follow_up:
                first = cache.majority_sample(member, rec["qid"])
                if first is None:
                    continue
                messages = second_turn(p, first["raw"])
            else:
                messages = p.messages
            rows.append({
                "prompt": [{"role": "system", "content": system_prompt(rec)}, *messages],
                "row_id": f"{rec['qid']}|{cond}",
                "qid": rec["qid"],
                "condition": cond,
                "pressure_target": p.pressure_target or "",  # Arrow-friendly: "" instead of None
            })
    return rows
