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
| inverse_surprising_popularity | answers + empirical second-order info over many questions | no |
| optimal_weight           | answers + member accuracies (OW-L / OW-I label-free, or a labelled calibration split) | OW-cal only |
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


# ---------------------------------------------------------------- ISP and OW
# Ai, Pan, Simchi-Levi, Tambe, Xu, "Beyond Majority Voting: LLM Aggregation by Leveraging
# Higher-Order Information", ICML 2026, arXiv 2510.01499v2. Equation numbers refer to that paper.
# Labels must be shuffled per question first (their Section 2); C1 already shuffles option order.

SecondOrder = dict[tuple[str, str], dict[tuple[str, str], float]]  # P[(i, j)][(s_k, s_l)] = P(A_i=s_k | A_j=s_l)


def second_order(questions: list[list[Output]], labels: list[str]) -> SecondOrder:
    """Empirical second-order information over M questions (Section 4.3):

        P^(A_i = s_k | A_j = s_l) = #{A_i = s_k, A_j = s_l} / #{A_j = s_l}

    counted over the questions where both i and j gave a valid answer. Conditionals on a label
    s_l that j never gave are left out; isp_advantage then uses 1/K for them (the "null
    information" value, Proposition E.1) and OW-L leaves them out of its loss.
    """
    votes = [{o["member"]: o["answer"] for o in valid_outputs(q)} for q in questions]
    members = sorted({m for v in votes for m in v})
    out: SecondOrder = {}
    for i in members:
        for j in members:
            if i == j:
                continue
            joint: Counter = Counter()
            marg: Counter = Counter()
            for v in votes:
                if i in v and j in v:
                    joint[(v[i], v[j])] += 1
                    marg[v[j]] += 1
            out[(i, j)] = {(k, l): joint[(k, l)] / marg[l] for k in labels for l in labels if marg[l]}
    return out


def isp_advantage(outputs: list[Output], P: SecondOrder, labels: list[str]) -> dict[str, float]:
    """Adv_ISP(s) = sum_i 1{a_i = s} - sum_i S_ISP(s, i)   (Eq. 5), with the ISP score (Eq. 4)

        S_ISP(s, i) = 1/(N-1) sum_{j != i} 1/(K-1) sum_{a in S without a_j} P(A_i = s | A_j = a).

    Only valid members take part (N = number of valid members).
    """
    votes = {o["member"]: o["answer"] for o in valid_outputs(outputs)}
    N, K = len(votes), len(labels)
    adv = {s: float(sum(a == s for a in votes.values())) for s in labels}
    if N < 2 or K < 2:
        return adv
    for s in labels:
        for i in votes:
            score = 0.0
            for j, a_j in votes.items():
                if j == i:
                    continue
                pij = P.get((i, j), {})
                vals = [pij.get((s, a), 1.0 / K) for a in labels if a != a_j]
                score += sum(vals) / (K - 1)
            adv[s] -= score / (N - 1)
    return adv


def inverse_surprising_popularity(
    outputs: list[Output], P: SecondOrder, labels: list[str], tie_key: str = ""
) -> str | None:
    """f_ISP = argmax_s Adv_ISP(s) (Algorithm 2). Fewer than 2 valid members -> majority vote."""
    if len(valid_outputs(outputs)) < 2:
        return majority_vote(outputs, tie_key)
    return pick_best(isp_advantage(outputs, P, labels), tie_key)


def clip_accuracy(x: float, K: int, eps: float = 1e-3) -> float:
    """Keep x in [1/K, 1 - eps]: 1/K gives weight 0 (agents at or below chance are excluded,
    as the paper says in Section 4); 1 - eps keeps the weight finite."""
    return min(1.0 - eps, max(1.0 / K, x))


def ow_weight(x: float, K: int, eps: float = 1e-3) -> float:
    """w_i = sigma_K^{-1}(x_i) with sigma_K(x) = e^x / (K - 1 + e^x)  =>  ln((K-1) x / (1 - x))."""
    x = clip_accuracy(x, K, eps)
    return math.log((K - 1) * x / (1 - x))


def optimal_weight(
    outputs: list[Output], accuracies: dict[str, float], K: int, tie_key: str = "", eps: float = 1e-3
) -> str | None:
    """f_OW = argmax_s sum_i w_i 1{a_i = s} (Algorithm 1). Members without an accuracy, or an
    all-zero weight vector, fall back to majority vote."""
    valid = valid_outputs(outputs)
    if not valid or any(o["member"] not in accuracies for o in valid):
        return majority_vote(outputs, tie_key)
    scores: dict[str, float] = defaultdict(float)
    for o in valid:
        scores[o["answer"]] += ow_weight(accuracies[o["member"]], K, eps)
    if all(abs(v) < 1e-12 for v in scores.values()):
        return majority_vote(outputs, tie_key)
    return pick_best(dict(scores), tie_key)


def model_second_order(xi: float, xj: float, same: bool, K: int) -> float:
    """P(A_i = s_k | A_j = s_l)[x] (Appendix F.2): same label if k = l, else different."""
    if same:
        return xi * xj + (1 - xi) * (1 - xj) / (K - 1)
    return (xi * (1 - xj) + (1 - xi) * xj) / (K - 1) + (K - 2) * (1 - xi) * (1 - xj) / (K - 1) ** 2


def ow_l_accuracies(
    P: SecondOrder, labels: list[str], steps: int = 3000, lr: float = 0.05, init: float = 0.7
) -> dict[str, float]:
    """OW-L (Eq. 6): accuracies minimising the squared gap between the model second-order
    information and the empirical one, summed over i != j and all (k, l). Solved by projected
    gradient descent on [1/K, 1] (the paper: "some gradient descent method")."""
    K = len(labels)
    members = sorted({m for pair in P for m in pair})
    if not members:
        return {}
    x = {m: max(1.0 / K, min(1.0, init)) for m in members}

    def grad(m: str) -> float:
        g = 0.0
        for (i, j), table in P.items():
            if m not in (i, j):
                continue
            for (k, l), p_hat in table.items():
                same = k == l
                r = model_second_order(x[i], x[j], same, K) - p_hat
                if same:
                    di = x[j] - (1 - x[j]) / (K - 1)
                    dj = x[i] - (1 - x[i]) / (K - 1)
                else:
                    di = (1 - 2 * x[j]) / (K - 1) - (K - 2) * (1 - x[j]) / (K - 1) ** 2
                    dj = (1 - 2 * x[i]) / (K - 1) - (K - 2) * (1 - x[i]) / (K - 1) ** 2
                if m == i:
                    g += 2 * r * di
                if m == j:
                    g += 2 * r * dj
        return g

    n_terms = max(1, sum(len(t) for t in P.values()))
    for _ in range(steps):
        g = {m: grad(m) / n_terms for m in members}
        x = {m: max(1.0 / K, min(1.0, x[m] - lr * g[m])) for m in members}
    return x


def ow_i_accuracies(questions: list[list[Output]], isp_answers: list[str | None]) -> dict[str, float]:
    """OW-I (Section 5.2): x_i = #{f_ISP = a_i} / M, with ISP's answer used as a pseudo label.
    M counts the questions where member i answered validly and ISP gave an answer."""
    hits: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for q, pseudo in zip(questions, isp_answers):
        if pseudo is None:
            continue
        for o in valid_outputs(q):
            counts[o["member"]] += 1
            hits[o["member"]] += int(o["answer"] == pseudo)
    return {m: hits[m] / counts[m] for m in counts}


def labelled_accuracies(questions: list[list[Output]], gold: list[str]) -> dict[str, float]:
    """Member accuracies on a labelled calibration split (the architecture doc's OW input).
    Invalid answers count as wrong."""
    hits: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for q, g in zip(questions, gold):
        for o in q:
            counts[o["member"]] += 1
            hits[o["member"]] += int(bool(o.get("valid")) and o.get("answer") == g)
    return {m: hits[m] / counts[m] for m in counts}


PER_QUESTION: dict[str, Callable[..., str | None]] = {
    "majority_vote": majority_vote,
    "confidence_weighted": confidence_weighted,
    "surprisingly_popular": surprisingly_popular,
}
