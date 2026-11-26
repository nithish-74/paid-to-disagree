"""C6: combined reward used by GRPO.

    R_i = l1*PP_cross + l2*BTS_info + l3*Pred + l4*Calib - l5*Flip   (+ l0*1[answer=gold], oracle only)

Every term is returned separately so we can see which one drives behaviour.

- PP_cross  Correlated Agreement vs. cached peer answers over the batch (rewards/ca.py)
- BTS_info  log(actual / predicted frequency), pooled own group + peers (rewards/bts.py)
- Pred      -KL(realised peer answers || own peer_prediction)             (rewards/bts.py)
- Calib     -(confidence - agreement target)^2   (negative Brier, so higher is better)
- Flip      1 if, under user_wrong / are_you_sure, the answer moves toward the user's view
            relative to this member's neutral majority answer. Never under user_right.

Gold labels: a task may carry `gold` ONLY when cfg.oracle is true; otherwise this module
raises. That is the code-level guarantee that labels never reach the training reward.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from rewards.bts import Dist, bts_info, prediction_score
from rewards.ca import PoolEntry, ca_batch

TERMS = ("pp_cross", "bts_info", "pred", "calib", "flip", "oracle")


@dataclass
class Report:
    """One parsed member output (own sample or cached peer report)."""

    answer: str | None
    peer_prediction: Dist | None = None
    confidence: float | None = None
    valid: bool = True

    @classmethod
    def from_output(cls, o: dict[str, Any]) -> "Report":
        return cls(o.get("answer"), o.get("peer_prediction"), o.get("confidence"), bool(o.get("valid")))

    @property
    def usable(self) -> bool:
        return self.valid and self.answer is not None


@dataclass
class RewardTask:
    qid: str
    condition: str
    keys: list[str]  # option letters for MCQ; [] for numeric answers
    samples: list[Report]  # the trained member's GRPO group for this prompt
    peers: list[Report]  # cached frozen peer reports, same peer order on every task
    pressure_target: str | None = None  # answer the pressure pushes toward (C2)
    neutral_majority: str | None = None  # this member's majority answer under neutral
    calib_targets: list[float] | None = None  # optional per-sample override
    gold: str | None = None  # oracle ablation ONLY


@dataclass
class RewardConfig:
    l1_pp_cross: float = 1.0
    l2_bts_info: float = 0.5
    l3_pred: float = 0.25
    l4_calib: float = 0.25
    l5_flip: float = 1.0
    l0_oracle: float = 1.0
    oracle: bool = False
    bts_mean: str = "geometric"
    eps: float = 1e-3
    # TODO(team): the doc does not fix these; tune on dev and report the values.
    # An unparseable completion scores (lowest valid total in its GRPO group) - invalid_margin,
    # so it always ranks last whatever the weights. A fixed penalty is not enough: a valid
    # sample can score about -5 at the start weights (CA -1, BTS_info ~ -2.3*0.5,
    # Pred ~ -6.9*0.25, Calib -0.25, Flip -1), and junk at -2 would then beat it.
    invalid_margin: float = 1.0
    invalid_penalty: float = -2.0  # only when no completion in the group is valid (advantage is 0 then)
    missing_pred: float = -1.0  # Pred when the sample gave no usable peer_prediction
    missing_confidence: float = -1.0  # Calib when the sample gave no confidence

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RewardConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown reward config keys: {sorted(unknown)}")
        return cls(**d)


def flip_indicator(condition: str, answer: str | None, pressure_target: str | None, neutral_majority: str | None) -> float:
    if answer is None or neutral_majority is None:
        return 0.0
    if condition == "user_wrong":
        return float(answer == pressure_target and neutral_majority != pressure_target)
    if condition == "are_you_sure":
        return float(answer != neutral_majority)
    return 0.0  # neutral, user_right (never penalised), peer_wrong (measured in eval only)


def calib_score(confidence: float | None, target: float, missing: float = -1.0) -> float:
    return missing if confidence is None else -((confidence - target) ** 2)


def default_calib_target(answer: str, task: RewardTask) -> float:
    """TODO(team): the doc says "confidence against agreement across conditions" without a
    precise target. Default: 1 if the answer matches this member's neutral majority answer
    (consistency across conditions); if that is unknown, the share of the member's group and
    peers that gave the same answer. Override per sample with task.calib_targets.
    """
    if task.neutral_majority is not None:
        return float(answer == task.neutral_majority)
    pooled = [r.answer for r in task.samples + task.peers if r.usable]
    return pooled.count(answer) / len(pooled) if pooled else 0.0


def combined_rewards(
    tasks: list[RewardTask], cfg: RewardConfig, seed: int = 0, pool: list[PoolEntry] | None = None
) -> list[list[dict[str, float]]]:
    """Reward terms and total for every sample: result[t][g] = {term: value, ..., "total", "invalid"}.

    `pool` (from the peer cache) supplies CA's different-question penalty pairs; see rewards/ca.py.
    """
    if not cfg.oracle and any(t.gold is not None for t in tasks):
        raise ValueError("gold labels reached the reward but oracle=false")
    rng = random.Random(seed)
    own = [[s.answer if s.usable else None for s in t.samples] for t in tasks]
    peers = [[p.answer if p.usable else None for p in t.peers] for t in tasks]
    pp_cross = ca_batch(own, peers, rng, pool=pool, qids=[t.qid for t in tasks] if pool is not None else None)

    out: list[list[dict[str, float]]] = []
    for ti, task in enumerate(tasks):
        pooled_answers = own[ti] + peers[ti]
        pooled_preds = [r.peer_prediction for r in task.samples + task.peers if r.usable and r.peer_prediction]
        row = []
        for gi, s in enumerate(task.samples):
            if not s.usable:  # total is set below, once the group's valid totals are known
                row.append({**{k: 0.0 for k in TERMS}, "total": cfg.invalid_penalty, "invalid": 1.0})
                continue
            pred = prediction_score(s.peer_prediction, peers[ti], task.keys, cfg.eps)
            target = task.calib_targets[gi] if task.calib_targets else default_calib_target(s.answer, task)
            terms = {
                "pp_cross": pp_cross[ti][gi],
                "bts_info": bts_info(s.answer, pooled_answers, pooled_preds, task.keys, cfg.bts_mean, cfg.eps),
                "pred": cfg.missing_pred if pred is None else pred,
                "calib": calib_score(s.confidence, target, cfg.missing_confidence),
                "flip": flip_indicator(task.condition, s.answer, task.pressure_target, task.neutral_majority),
                "oracle": float(cfg.oracle and task.gold is not None and s.answer == task.gold),
            }
            total = (
                cfg.l1_pp_cross * terms["pp_cross"]
                + cfg.l2_bts_info * terms["bts_info"]
                + cfg.l3_pred * terms["pred"]
                + cfg.l4_calib * terms["calib"]
                - cfg.l5_flip * terms["flip"]
                + cfg.l0_oracle * terms["oracle"]
            )
            row.append({**terms, "total": total, "invalid": 0.0})
        valid_totals = [r["total"] for r in row if not r["invalid"]]
        floor = min(valid_totals) - cfg.invalid_margin if valid_totals else cfg.invalid_penalty
        for r in row:
            if r["invalid"]:
                r["total"] = floor
        out.append(row)
    return out


def summarize(rewards: list[list[dict[str, float]]]) -> dict[str, float]:
    """Mean of every term over all samples (for per-step logging)."""
    flat = [r for row in rewards for r in row]
    if not flat:
        return {}
    return {k: sum(r[k] for r in flat) / len(flat) for k in (*TERMS, "total", "invalid")}


def config_dict(cfg: RewardConfig) -> dict[str, Any]:
    return asdict(cfg)
