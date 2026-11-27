"""PP_cross via the Peer Truth Serum (PTS): needs no prediction report.

Source: Faltings, Jurca, Radanovic, "Peer Truth Serum: Incentives for Crowdsourcing
Measurements and Opinions", arXiv 1704.05269, Section 8, Definition 17, Eq. 5:

    tau(r, rr, R) = f(rr) + { C / R[r]   if r = rr
                            { 0          otherwise

r = the agent's report, rr = a peer's reference report, R = public distribution of reports
(normalised histogram, initialised fully mixed, e.g. all ones; Section 4.1). C > 0 does not
depend on the reports; f(rr) does not depend on r. f(rr) = -C makes the expected payment 0
for an agent who only knows R (the paper's arbitrage-free setting).

Project choices (not fixed by the paper): rr = each peer's answer on the same question with
the payment averaged over peers; R = histogram of all peers' answers over the batch and the
cached pool (other questions), started from ones; C = 1, f = -1.
Note: option letters are shuffled per question, so R is close to uniform and PTS is then close
to K * agreement - 1.
"""

from __future__ import annotations

from collections import Counter

from rewards.ca import Answer, PoolEntry


def pts_score(r: Answer, rr: Answer, R: dict[str, float], C: float = 1.0, f: float = -1.0) -> float:
    """tau(r, rr, R) = f(rr) + C / R[r] if r == rr, else f(rr)   (Definition 17)."""
    if C <= 0:
        raise ValueError("C must be > 0")
    if r is not None and r == rr:
        if R.get(r, 0.0) <= 0:
            raise ValueError(f"R[{r!r}] must be > 0")
        return f + C / R[r]
    return f


def report_distribution(answers: list[Answer], keys: list[str], prior_count: float = 1.0) -> dict[str, float]:
    """R: normalised histogram of reports over `keys`, every count started at `prior_count`."""
    counts = Counter(a for a in answers if a in keys)
    total = sum(counts.values()) + prior_count * len(keys)
    return {k: (counts[k] + prior_count) / total for k in keys}


def pts_batch(
    own: list[list[Answer]],
    peers: list[list[Answer]],
    keys: list[list[str]],
    pool: list[PoolEntry] | None = None,
    C: float = 1.0,
    f: float = -1.0,
) -> list[list[float]]:
    """PTS for every own sample: scores[t][g], averaged over the peers with a valid answer.

    R for task t is built from every peer answer in this batch and the pool that fits task t's
    keys. Numeric tasks (keys == []) score 0.
    """
    if not (len(peers) == len(keys) == len(own)):
        raise ValueError("own, peers and keys must cover the same tasks")
    peer_answers = [a for row in peers for a in row] + [a for e in (pool or []) for a in e[2]]
    scores: list[list[float]] = []
    for t in range(len(own)):
        if len(keys[t]) < 2:
            scores.append([0.0] * len(own[t]))
            continue
        R = report_distribution(peer_answers, keys[t])
        refs = [p for p in peers[t] if p is not None]
        row = []
        for a in own[t]:
            row.append(sum(pts_score(a, rr, R, C, f) for rr in refs) / len(refs) if refs else 0.0)
        scores.append(row)
    return scores
