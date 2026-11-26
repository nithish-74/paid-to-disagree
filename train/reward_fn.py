"""C7 step 3: the reward function handed to TRL GRPOTrainer.

TRL calls reward(prompts, completions, **dataset_columns) with every prompt repeated once
per generation. We group completions by row_id (one GRPO group per prompt), parse them,
attach the frozen peers' cached reports and call rewards/combined.py. Every call's term
means are appended to reward_terms.jsonl so we can see which term drives behaviour.

Gold labels: question records are copied WITHOUT "gold" before the reward ever sees them.
Golds are only passed in when the reward config has oracle=true (oracle ablation).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from council.members import parse_answer
from data.build_questions import LETTERS, Record
from rewards.combined import RewardConfig, RewardTask, Report, combined_rewards, summarize
from train.peer_cache import PeerCache


def strip_gold(records: dict[str, Record]) -> dict[str, Record]:
    return {qid: {k: v for k, v in rec.items() if k != "gold"} for qid, rec in records.items()}


def completion_text(completion: Any) -> str:
    """TRL passes str (standard) or [{"role": "assistant", "content": ...}] (conversational)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def make_reward_fn(
    cache: PeerCache,
    member: str,
    records: dict[str, Record],
    cfg: RewardConfig,
    log_path: Path | None = None,
    seed: int = 0,
    golds: dict[str, str] | None = None,
    population: str = "cross_model",
) -> Callable[..., list[float]]:
    """population: "cross_model" (peers = the other frozen members) or "single_model"
    (peers = frozen copies of this member, baseline B9); see train/peer_cache.py."""
    if golds and not cfg.oracle:
        raise ValueError("golds given but oracle=false: labels must not reach the reward")
    safe = strip_gold(records)
    pool = cache.pool(member, population)
    state = {"calls": 0}

    def peer_prediction_reward(prompts, completions, row_id, qid, condition, pressure_target, **kwargs) -> list[float]:
        groups: OrderedDict[str, list[int]] = OrderedDict()
        for i, rid in enumerate(row_id):
            groups.setdefault(rid, []).append(i)

        tasks = []
        for rid, idxs in groups.items():
            i0 = idxs[0]
            rec = safe[qid[i0]]
            samples = []
            for i in idxs:
                parsed = parse_answer(completion_text(completions[i]), rec)
                samples.append(Report(parsed.answer, parsed.peer_prediction, parsed.confidence, parsed.error is None))
            keys = list(LETTERS[: len(rec["options"])]) if rec["answer_type"] == "mcq" else []
            tasks.append(RewardTask(
                qid=qid[i0],
                condition=condition[i0],
                keys=keys,
                samples=samples,
                peers=cache.peers_for(member, qid[i0], population),
                pressure_target=pressure_target[i0] or None,
                neutral_majority=cache.neutral_majority(member, qid[i0]),
                gold=golds.get(qid[i0]) if (cfg.oracle and golds) else None,
            ))

        rewards = combined_rewards(tasks, cfg, seed=seed + state["calls"], pool=pool)
        flat = [0.0] * len(completions)
        for idxs, row in zip(groups.values(), rewards):
            for i, r in zip(idxs, row):
                flat[i] = r["total"]

        state["calls"] += 1
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"call": state["calls"], "member": member, "n": len(flat),
                     "conditions": sorted(set(condition)), **summarize(rewards)}
            with Path(log_path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        return flat

    return peer_prediction_reward
