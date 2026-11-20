"""Bayesian-Truth-Serum style terms: information score (BTS_info) and prediction score (Pred).

From the architecture doc, C6:
  BTS_info = log( actual frequency of the chosen answer / its predicted frequency ),
             pooled over the member's own GRPO group and the peers.
             Rewards answers that are more common than expected (how a correct minority gets paid).
  Pred     = -KL( realised peer answer distribution || member's peer_prediction ).

TODO(team): the doc does not say how predictions are averaged into the "predicted
frequency". Default is the geometric mean (Prelec 2004, original BTS); `mean="arithmetic"`
is available. Confirm against Mytsyk et al. 2026 and set it in configs/reward.yaml.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

Dist = dict[str, float]


def support(keys: Iterable[str], answers: Iterable[str | None], preds: Iterable[Dist]) -> list[str]:
    """All outcomes in play: option letters, given answers and predicted values (for numbers)."""
    out = set(keys) | {a for a in answers if a is not None}
    for p in preds:
        out |= set(p)
    return sorted(out)


def smooth(dist: Dist, keys: Iterable[str], eps: float) -> Dist:
    """Add eps to every key and renormalise, so logs and KL stay finite."""
    keys = list(keys)
    raw = {k: dist.get(k, 0.0) + eps for k in keys}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


def actual_frequency(answers: list[str | None]) -> Dist:
    valid = [a for a in answers if a is not None]
    if not valid:
        return {}
    return {k: n / len(valid) for k, n in Counter(valid).items()}


def predicted_frequency(preds: list[Dist], keys: list[str], mean: str = "geometric", eps: float = 1e-3) -> Dist:
    """Average of (smoothed) predictions over `keys`; geometric means are renormalised."""
    if not preds:
        return {}
    smoothed = [smooth(p, keys, eps) for p in preds]
    if mean == "arithmetic":
        return {k: sum(p[k] for p in smoothed) / len(smoothed) for k in keys}
    if mean != "geometric":
        raise ValueError(f"mean must be 'geometric' or 'arithmetic', got {mean!r}")
    geo = {k: math.exp(sum(math.log(p[k]) for p in smoothed) / len(smoothed)) for k in keys}
    z = sum(geo.values())
    return {k: v / z for k, v in geo.items()}


def bts_info(
    answer: str | None,
    pooled_answers: list[str | None],
    pooled_preds: list[Dist],
    keys: list[str],
    mean: str = "geometric",
    eps: float = 1e-3,
) -> float:
    """log(xbar[answer] / ybar[answer]); 0 if the answer is invalid or no predictions exist."""
    if answer is None or not pooled_preds:
        return 0.0
    xbar = actual_frequency(pooled_answers)
    ybar = predicted_frequency(pooled_preds, support(keys, pooled_answers, pooled_preds), mean, eps)
    return math.log(max(xbar.get(answer, 0.0), eps) / max(ybar.get(answer, 0.0), eps))


def kl(p: Dist, q: Dist) -> float:
    """KL(p || q) over p's support; q must be smoothed (strictly positive on p's support)."""
    return sum(pk * math.log(pk / q[k]) for k, pk in p.items() if pk > 0)


def prediction_score(prediction: Dist | None, peer_answers: list[str | None], keys: list[str], eps: float = 1e-3) -> float | None:
    """-KL(realised peer distribution || prediction). None if it cannot be computed."""
    realised = actual_frequency(peer_answers)
    if prediction is None or not realised:
        return None
    return -kl(realised, smooth(prediction, support(keys, peer_answers, [prediction]), eps))
