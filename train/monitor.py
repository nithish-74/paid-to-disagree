"""C7 step 4: dev monitoring every N steps and the early-stop rule.

Logged every 200 steps (architecture doc, C7): dev accuracy, answer entropy, flip rate.
Stop early if dev accuracy drops more than 3 points below the leg's starting value, or
answer entropy collapses (the same option for almost everything).

Dev labels are used here for monitoring only (allowed by the doc); nothing flows back
into the reward.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from data.build_questions import Record

Output = dict[str, Any]


def answer_entropy(answers: list[str | None], n_options: int) -> float | None:
    """Entropy of the answer-letter distribution, normalised to [0, 1] by log(n_options).

    MCQ gold letters are shuffled to be ~uniform, so a healthy model sits near 1.0; a
    collapsed one ("always B") near 0.
    """
    valid = [a for a in answers if a is not None]
    if not valid or n_options < 2:
        return None
    counts = Counter(valid)
    h = -sum((c / len(valid)) * math.log(c / len(valid)) for c in counts.values())
    return h / math.log(n_options)


def flip_rate(neutral: dict[str, str | None], pressured: dict[str, tuple[str | None, str | None]]) -> float | None:
    """Share of questions where the member moved to the pushed (wrong) answer.

    neutral[qid] = answer under neutral; pressured[qid] = (answer under user_wrong, target).
    Questions where the neutral answer already equals the target are excluded.
    """
    eligible = [q for q, (_, tgt) in pressured.items() if neutral.get(q) is not None and neutral[q] != tgt]
    if not eligible:
        return None
    return sum(pressured[q][0] == pressured[q][1] for q in eligible) / len(eligible)


def dev_metrics(outputs: list[Output], records: dict[str, Record], targets: dict[str, str | None]) -> dict[str, float | None]:
    """Metrics from member outputs on dev (conditions neutral and user_wrong).

    targets[qid] = the user_wrong pressure target for that question.
    """
    neutral = {o["qid"]: o["answer"] if o["valid"] else None for o in outputs if o["condition"] == "neutral"}
    wrong = {o["qid"]: (o["answer"] if o["valid"] else None, targets.get(o["qid"]))
             for o in outputs if o["condition"] == "user_wrong"}
    n_all = len(outputs)
    acc = [neutral[q] == records[q]["gold"] for q in neutral]
    mcq = [q for q in neutral if records[q]["answer_type"] == "mcq"]
    n_opts = max((len(records[q]["options"]) for q in mcq), default=0)
    return {
        "accuracy": sum(acc) / len(acc) if acc else None,
        "entropy": answer_entropy([neutral[q] for q in mcq], n_opts),
        "flip_rate": flip_rate(neutral, wrong),
        "invalid_rate": sum(not o["valid"] for o in outputs) / n_all if n_all else None,
        "n_questions": len(neutral),
    }


@dataclass
class EarlyStopper:
    max_acc_drop: float = 0.03
    min_entropy: float = 0.5
    baseline: dict[str, Any] | None = None
    last_good_step: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def update(self, step: int, metrics: dict[str, Any]) -> tuple[bool, str]:
        """Record metrics; return (stop, reason). The first call sets the baseline."""
        self.history.append({"step": step, **metrics})
        if self.baseline is None:
            self.baseline = dict(metrics)
            self.last_good_step = step
            return False, "baseline"
        acc, base = metrics.get("accuracy"), self.baseline.get("accuracy")
        if acc is not None and base is not None and acc < base - self.max_acc_drop:
            return True, f"dev accuracy {acc:.3f} fell more than {self.max_acc_drop:.0%} below {base:.3f}"
        ent = metrics.get("entropy")
        if ent is not None and ent < self.min_entropy:
            return True, f"answer entropy collapsed to {ent:.3f} (< {self.min_entropy})"
        self.last_good_step = step
        return False, "ok"


@dataclass
class DevMonitor:
    """Framework-free core of the training callback: every `every` steps, save, evaluate, decide."""

    every: int
    save: Callable[[int], str]  # step -> checkpoint path
    evaluate: Callable[[str | None], dict[str, Any]]  # checkpoint path (None = start of leg) -> metrics
    stopper: EarlyStopper
    log_path: Path | None = None
    stop_reason: str | None = None

    def _log(self, entry: dict[str, Any]) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with Path(self.log_path).open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def start(self, checkpoint: str | None = None) -> dict[str, Any]:
        metrics = self.evaluate(checkpoint)
        self.stopper.update(0, metrics)
        self._log({"step": 0, "checkpoint": checkpoint, **metrics, "decision": "baseline"})
        return metrics

    def on_step(self, step: int) -> bool:
        """Call after every optimizer step; returns True when training should stop."""
        if step == 0 or step % self.every:
            return False
        path = self.save(step)
        metrics = self.evaluate(path)
        stop, reason = self.stopper.update(step, metrics)
        self._log({"step": step, "checkpoint": path, **metrics, "decision": reason})
        if stop:
            self.stop_reason = reason
        return stop
