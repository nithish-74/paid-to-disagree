"""C7 step 1: cached reports from the council, so peers stay frozen during a training leg.

For every training question, each member answers under `neutral` n_samples times
(temperature > 0). From those samples the cache provides:
  - a peer report per member (the first sample that gives the member's majority answer)
  - each member's neutral majority answer (the reference for the Flip term)
  - a pool of (qid, own answer, peer answers) for CA's different-question penalty

Peers answer under neutral on purpose: the reward compares a pressured member against
what its peers honestly think, not against peers under the same pressure.

On disk: JSONL, one line per (member, qid): {"member", "qid", "samples": [MemberOutput dict, ...]}.
Refreshing a member after its training leg replaces only that member's lines.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from council.members import CouncilMember
from data.build_questions import Record
from data.pressure import apply_condition
from rewards.ca import PoolEntry
from rewards.combined import Report

Sample = dict[str, Any]


class PeerCache:
    def __init__(self, entries: dict[tuple[str, str], list[Sample]] | None = None) -> None:
        self.entries: dict[tuple[str, str], list[Sample]] = entries or {}

    # ---------------------------------------------------------------- access

    @property
    def members(self) -> list[str]:
        return sorted({m for m, _ in self.entries})

    @property
    def qids(self) -> list[str]:
        return sorted({q for _, q in self.entries})

    def samples(self, member: str, qid: str) -> list[Sample]:
        return self.entries.get((member, qid), [])

    def majority(self, member: str, qid: str) -> str | None:
        answers = [s["answer"] for s in self.samples(member, qid) if s.get("valid") and s.get("answer") is not None]
        if not answers:
            return None
        counts = Counter(answers)
        top = max(counts.values())
        # deterministic tie-break: first sample (in generation order) with a top answer
        return next(a for a in answers if counts[a] == top)

    def majority_sample(self, member: str, qid: str) -> Sample | None:
        maj = self.majority(member, qid)
        return next((s for s in self.samples(member, qid) if s.get("answer") == maj and s.get("valid")), None)

    def report(self, member: str, qid: str) -> Report:
        s = self.majority_sample(member, qid)
        return Report.from_output(s) if s else Report(None, valid=False)

    def peer_names(self, member: str) -> list[str]:
        return [m for m in self.members if m != member]

    def peers_for(self, member: str, qid: str) -> list[Report]:
        """Frozen peers' reports, always in the same (sorted) peer order."""
        return [self.report(p, qid) for p in self.peer_names(member)]

    def neutral_majority(self, member: str, qid: str) -> str | None:
        return self.majority(member, qid)

    def pool(self, member: str) -> list[PoolEntry]:
        peers = self.peer_names(member)
        return [(q, self.majority(member, q), [self.majority(p, q) for p in peers]) for q in self.qids]

    # ---------------------------------------------------------------- update / io

    def update_member(self, member: str, samples_by_qid: dict[str, list[Sample]]) -> None:
        self.entries = {k: v for k, v in self.entries.items() if k[0] != member}
        for qid, samples in samples_by_qid.items():
            self.entries[(member, qid)] = samples

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for (member, qid), samples in sorted(self.entries.items()):
                f.write(json.dumps({"member": member, "qid": qid, "samples": samples}, ensure_ascii=False) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PeerCache":
        entries = {}
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    entries[(d["member"], d["qid"])] = d["samples"]
        return cls(entries)


def generate_samples(member: CouncilMember, records: list[Record], n_samples: int, seed: int = 0) -> dict[str, list[Sample]]:
    """n_samples neutral answers per question from one member (one batched call per sample index)."""
    recs = {r["qid"]: r for r in records}
    prompts = [apply_condition(r, "neutral") for r in records]
    out: dict[str, list[Sample]] = {r["qid"]: [] for r in records}
    for k in range(n_samples):
        member.seed = seed + k
        for turns in member.answer_prompts(prompts, recs, stage="cache"):
            o = turns[-1]
            out[o.qid].append(o.to_dict())
    return out
