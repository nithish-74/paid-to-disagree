"""PP_cross via Correlated Agreement (multi-task peer prediction).

Definition used (architecture doc, C6): agreement on the same question minus agreement on
randomly paired DIFFERENT questions. For member i's sample on task t and peer j:

    score = 1[a_i(t) == a_j(t)] - 1[a_i(t1) == a_j(t2)],   t, t1, t2 all distinct

averaged over peers. Answering the same thing everywhere earns 0 in expectation, which is
what stops "everyone gives the same answer" from paying. Fully label-free.
"""

from __future__ import annotations

import random

Answer = str | None


def agree(a: Answer, b: Answer) -> float:
    return 1.0 if a is not None and a == b else 0.0


def ca_score(own_t: Answer, peer_t: Answer, own_other: Answer, peer_other: Answer) -> float:
    return agree(own_t, peer_t) - agree(own_other, peer_other)


PoolEntry = tuple[str, Answer, list[Answer]]  # (qid, own answer, peer answers in peer order)


def _pool_pair(pool: list[PoolEntry], qid: str | None, rng: random.Random, tries: int = 20) -> tuple[int, int] | None:
    """Two distinct pool indices whose qids differ from each other and from `qid`."""
    if len(pool) < 3:
        return None
    for _ in range(tries):
        i1, i2 = rng.sample(range(len(pool)), 2)
        if pool[i1][0] != qid and pool[i2][0] != qid and pool[i1][0] != pool[i2][0]:
            return i1, i2
    return None


def ca_batch(
    own: list[list[Answer]],
    peers: list[list[Answer]],
    rng: random.Random,
    pool: list[PoolEntry] | None = None,
    qids: list[str] | None = None,
) -> list[list[float]]:
    """CA score for every own sample.

    own[t][g]   member i's g-th sampled answer on task t (the GRPO group)
    peers[t][j] peer j's cached answer on task t (same peer order on every task)
    pool        optional (qid, own answer, peer answers) from the cache over the whole
                training set. When given, the penalty pairs t1, t2 are drawn from it, so CA
                works even when a GRPO step only sees 1-2 questions (the usual case on a
                T4). The own answer in the pool is the member's cached (pre-update) answer.
    Returns scores[t][g]. Without a pool, needs T >= 3 tasks; with fewer, the penalty term
    is skipped (score = plain agreement) -- avoid that in training.
    """
    T = len(own)
    if len(peers) != T:
        raise ValueError("own and peers must cover the same tasks")
    if pool is not None and (qids is None or len(qids) != T):
        raise ValueError("pool requires one qid per task")
    scores: list[list[float]] = []
    for t in range(T):
        row = []
        others = [x for x in range(T) if x != t]
        for a in own[t]:
            per_peer = []
            for j, peer_ans in enumerate(peers[t]):
                pair = _pool_pair(pool, qids[t], rng) if pool is not None else None
                if pair is not None:
                    own_other = pool[pair[0]][1]
                    peer_list = pool[pair[1]][2]
                    penalty = agree(own_other, peer_list[j] if j < len(peer_list) else None)
                elif pool is None and len(others) >= 2:
                    t1, t2 = rng.sample(others, 2)
                    own_other = rng.choice(own[t1]) if own[t1] else None
                    penalty = agree(own_other, peers[t2][j])
                else:
                    penalty = 0.0
                per_peer.append(agree(a, peer_ans) - penalty)
            row.append(sum(per_peer) / len(per_peer) if per_peer else 0.0)
        scores.append(row)
    return scores
