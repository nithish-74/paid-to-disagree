"""C5: aggregators. Each takes member outputs for one question and returns one answer.

Inputs are MemberOutput dicts (council/members.py). Invalid outputs are ignored.
Ties are broken with an RNG seeded by `tie_key` (use the qid), never by letter order.

| Aggregator               | Uses                          | Labels |
|--------------------------|-------------------------------|--------|
| majority_vote            | answers                       | no     |
| confidence_weighted      | answers + confidence          | no     |
| surprisingly_popular     | answers + peer_prediction     | no     |
| dawid_skene              | answers across many questions | no     |
| chairman                 | Karpathy stage-3 output       | no     |
| isp / optimal_weight     | STUBS: equations from Ai et al. (arXiv 2510.01499) to be pasted |
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable

from data.build_questions import seeded_rng

Output = dict[str, Any]


def valid_outputs(outputs: list[Output]) -> list[Output]:
    return [o for o in outputs if o.get("valid") and o.get("answer") is not None]


def pick_best(scores: dict[str, float], tie_key: str = "", tol: float = 1e-12) -> str | None:
    """Highest-scoring key; ties broken by a seeded RNG so no option letter is favoured."""
    if not scores:
        return None
    top = max(scores.values())
    tied = sorted(k for k, v in scores.items() if v >= top - tol)
    return tied[0] if len(tied) == 1 else seeded_rng("tie", tie_key).choice(tied)


def majority_vote(outputs: list[Output], tie_key: str = "") -> str | None:
    return pick_best(dict(Counter(o["answer"] for o in valid_outputs(outputs))), tie_key)


def confidence_weighted(outputs: list[Output], tie_key: str = "", missing_confidence: float = 0.5) -> str | None:
    """Sum of stated confidences per answer; a missing confidence counts as `missing_confidence`."""
    scores: dict[str, float] = defaultdict(float)
    for o in valid_outputs(outputs):
        conf = o.get("confidence")
        scores[o["answer"]] += missing_confidence if conf is None else conf
    return pick_best(dict(scores), tie_key)


def surprisingly_popular(outputs: list[Output], tie_key: str = "") -> str | None:
    """Pick the answer whose actual share most exceeds its predicted share (actual - predicted).

    Actual share: fraction of valid members giving the answer.
    Predicted share: mean of the members' peer_prediction for it.
    Falls back to majority vote when no member gave a usable peer_prediction.
    """
    valid = valid_outputs(outputs)
    if not valid:
        return None
    preds = [o["peer_prediction"] for o in valid if o.get("peer_prediction")]
    if not preds:
        return majority_vote(outputs, tie_key)
    actual = Counter(o["answer"] for o in valid)
    candidates = set(actual) | {k for p in preds for k in p}
    scores = {
        k: actual.get(k, 0) / len(valid) - sum(p.get(k, 0.0) for p in preds) / len(preds) for k in candidates
    }
    return pick_best(scores, tie_key)


def dawid_skene(
    questions: list[list[Output]],
    n_classes: list[int | None] | None = None,
    tie_keys: list[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-3,
) -> tuple[list[str | None], dict[str, float]]:
    """One-coin Dawid-Skene: EM over many questions with one accuracy per member.

    One-coin (not a full confusion matrix) because MCQ letters are shuffled per question,
    so letter-to-letter confusions carry no meaning; it also works for numeric answers.
    Errors are spread evenly over the other (n_classes - 1) classes; n_classes defaults to
    max(2, number of distinct answers given). Returns (answers, member accuracies).
    """
    n_classes = n_classes or [None] * len(questions)
    tie_keys = tie_keys or [str(i) for i in range(len(questions))]
    votes = [{o["member"]: o["answer"] for o in valid_outputs(q)} for q in questions]
    members = sorted({m for v in votes for m in v})

    # Init posteriors with soft majority vote.
    posts: list[dict[str, float]] = []
    for v in votes:
        c = Counter(v.values())
        total = sum(c.values())
        posts.append({k: n / total for k, n in c.items()} if total else {})

    acc = {m: 0.7 for m in members}
    for _ in range(max_iter):
        # M-step: expected accuracy per member.
        hits: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for v, post in zip(votes, posts):
            for m, a in v.items():
                hits[m] += post.get(a, 0.0)
                counts[m] += 1
        new_acc = {m: min(1 - eps, max(eps, hits[m] / counts[m])) for m in members}
        # E-step: posterior over the answers given for each question.
        new_posts = []
        for v, k_override in zip(votes, n_classes):
            cands = sorted(set(v.values()))
            if not cands:
                new_posts.append({})
                continue
            k = max(2, k_override or len(cands))
            logp = {
                c: sum(math.log(new_acc[m]) if a == c else math.log((1 - new_acc[m]) / (k - 1)) for m, a in v.items())
                for c in cands
            }
            top = max(logp.values())
            z = sum(math.exp(lp - top) for lp in logp.values())
            new_posts.append({c: math.exp(lp - top) / z for c, lp in logp.items()})
        delta = max((abs(new_acc[m] - acc[m]) for m in members), default=0.0)
        acc, posts = new_acc, new_posts
        if delta < tol:
            break
    answers = [pick_best(p, key) for p, key in zip(posts, tie_keys)]
    return answers, acc


def chairman(result: dict[str, Any]) -> str | None:
    """Final answer of the Karpathy-style chairman (ProtocolResult.to_dict())."""
    stage = result.get("stages", {}).get("stage3_chairman") or []
    return stage[0]["answer"] if stage and stage[0].get("valid") else None


def inverse_surprising_popularity(outputs: list[Output], **kwargs: Any) -> str | None:
    raise NotImplementedError(
        "ISP (Ai et al., ICML 2026, arXiv 2510.01499): paste the paper's equations before implementing."
    )


def optimal_weight(outputs: list[Output], calibration: Any = None, **kwargs: Any) -> str | None:
    raise NotImplementedError(
        "OW (Ai et al., ICML 2026, arXiv 2510.01499): paste the paper's equations before implementing. "
        "Needs a small calibration split."
    )


PER_QUESTION: dict[str, Callable[..., str | None]] = {
    "majority_vote": majority_vote,
    "confidence_weighted": confidence_weighted,
    "surprisingly_popular": surprisingly_popular,
}
