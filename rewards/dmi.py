"""PP_cross via Determinant Mutual Information (DMI) multi-task peer prediction.

Source: Kong, "Dominantly Truthful Multi-task Peer Prediction with a Constant Number of
Tasks", SODA 2020, arXiv 1911.00272, Section 5 (DMI-Mechanism, Theorem 5.1). Chen et al.,
"Incentivizing Truthful Language Models via Peer Elicitation Games" (arXiv 2505.13636,
Section 2.2) use the same payment with C = 2.

n agents answer T >= 2C a-priori similar tasks with C choices. The tasks are split into two
disjoint parts T1, T2 with |T1|, |T2| >= C. For agents i != j and l = 1, 2:

    M_l^{ij}(c, c') = sum_{t in T_l} 1((c_i^t, c_j^t) = (c, c'))          (C x C counts)
    p_i = sum_{j != i} det(M_1^{ij}) det(M_2^{ij})
    q_i = p_i / ((n - 1) (C!)^2 binom(|T1|, C) binom(|T2|, C))           (normalised)

Expected q_i is in [0, (1/C)^C], and answering the same thing everywhere earns 0.

Use as a GRPO reward (project choice, not in the paper): a GRPO step sees only 1-2 questions,
so the other T - 1 tasks come from the cached pool (as in rewards/ca.py). For sample g of
member i on question t, q_i is computed with i's answer on t set to that sample and i's
cached answers on the pool tasks; peers' cached answers throughout. Tasks where i or a peer
has no valid answer are left out of that pair's matrices.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from rewards.ca import Answer, PoolEntry


def det(matrix: list[list[int]]) -> int:
    """Exact integer determinant (Gaussian elimination over fractions)."""
    m = [[Fraction(v) for v in row] for row in matrix]
    n = len(m)
    sign = 1
    result = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            return 0
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            sign = -sign
        result *= m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            if factor:
                m[r] = [a - factor * b for a, b in zip(m[r], m[col])]
    return int(sign * result)


def answer_matrix(pairs: list[tuple[Answer, Answer]], keys: list[str]) -> list[list[int]]:
    """M(c, c') = number of tasks where agent i said c and agent j said c'. Pairs with a None
    or an answer outside `keys` are skipped."""
    index = {k: n for n, k in enumerate(keys)}
    m = [[0] * len(keys) for _ in keys]
    for a, b in pairs:
        if a in index and b in index:
            m[index[a]][index[b]] += 1
    return m


def dmi_payment(
    own: list[Answer], peers: list[list[Answer]], keys: list[str], split: int, normalise: bool = True
) -> float:
    """DMI payment for one agent over T tasks.

    own[t]       the agent's answer on task t
    peers[t][j]  peer j's answer on task t (same peer order on every task)
    split        tasks [0, split) form T1 and [split, T) form T2
    """
    T, C = len(own), len(keys)
    if len(peers) != T:
        raise ValueError("own and peers must cover the same tasks")
    t1, t2 = split, T - split
    if t1 < C or t2 < C:
        raise ValueError(f"DMI needs |T1|, |T2| >= C = {C}; got {t1}, {t2}")
    n_peers = max((len(p) for p in peers), default=0)
    if n_peers == 0:
        return 0.0
    p = 0
    for j in range(n_peers):
        pairs = [(own[t], peers[t][j] if j < len(peers[t]) else None) for t in range(T)]
        p += det(answer_matrix(pairs[:split], keys)) * det(answer_matrix(pairs[split:], keys))
    if not normalise:
        return float(p)
    return p / (n_peers * math.factorial(C) ** 2 * math.comb(t1, C) * math.comb(t2, C))


def dmi_batch(
    own: list[list[Answer]],
    peers: list[list[Answer]],
    keys: list[list[str]],
    rng: random.Random,
    pool: list[PoolEntry],
    qids: list[str],
    n_tasks: int = 16,
) -> list[list[float]]:
    """DMI score for every own sample: scores[t][g].

    own[t][g]    member i's g-th sampled answer on task t (the GRPO group)
    peers[t][j]  peer j's cached answer on task t
    keys[t]      option letters of task t (C = len(keys[t]); numeric tasks have [] and score 0)
    pool         (qid, own cached answer, peer cached answers) over the training set; the
                 n_tasks - 1 other tasks are drawn from pool entries with a different qid
                 whose answers all fit task t's keys
    n_tasks      T >= 2C; the T tasks are split into random halves (seeded by rng)
    """
    if not (len(peers) == len(keys) == len(qids) == len(own)):
        raise ValueError("own, peers, keys and qids must cover the same tasks")
    scores: list[list[float]] = []
    for t in range(len(own)):
        C = len(keys[t])
        if C < 2:
            scores.append([0.0] * len(own[t]))
            continue
        if n_tasks < 2 * C:
            raise ValueError(f"n_tasks must be >= 2C = {2 * C}")
        allowed = set(keys[t]) | {None}
        candidates = [e for e in pool if e[0] != qids[t] and e[1] in allowed and all(a in allowed for a in e[2])]
        if len(candidates) < n_tasks - 1:
            raise ValueError(f"pool has {len(candidates)} usable tasks; DMI needs {n_tasks - 1}")
        others = rng.sample(candidates, n_tasks - 1)
        order = list(range(n_tasks))  # position 0 is the current task
        rng.shuffle(order)
        split = n_tasks // 2
        row = []
        for a in own[t]:
            own_col: list[Answer] = [a] + [e[1] for e in others]
            peer_col: list[list[Answer]] = [list(peers[t])] + [list(e[2]) for e in others]
            row.append(dmi_payment([own_col[k] for k in order], [peer_col[k] for k in order], keys[t], split))
        scores.append(row)
    return scores
