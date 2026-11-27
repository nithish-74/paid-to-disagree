"""C8: turn runs/<run_id>/members.jsonl into the paper's tables. No metric is computed by hand.

Tables (CSV in runs/<run_id>/metrics/, plus metrics.json):
  members.csv      per cell x member: accuracy, invalid rate, ECE, Brier, answer entropy,
                   prediction KL (peer_prediction vs realised peer answers)
  flips.csv        per pressured cell x member: flip-to-wrong, flip-to-right, cave rate
                   (moved to the pushed answer; under peer_wrong this is the peer-flip rate)
  pairwise.csv     per cell x member pair: error correlation (phi), Cohen's kappa   [H3]
  aggregation.csv  per cell x aggregator: accuracy and gain over majority vote      [H4]
                   (MCQ cells also get isp, ow_l, ow_i, and ow_cal with --ow-calibration)
  ow_calibration.json  per dataset x member stage-1 neutral accuracy; from a dev-split run
                   this is the labelled calibration input for ow_cal

A cell is (dataset, protocol, condition). Every number carries a 95% bootstrap CI from
resampling questions. Run 3 seeds and combine them in analysis/ before claiming anything.

    python -m eval.metrics runs/eval_untrained_seed0 [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from council.aggregators import (
    chairman,
    confidence_weighted,
    dawid_skene,
    inverse_surprising_popularity,
    labelled_accuracies,
    majority_vote,
    optimal_weight,
    ow_i_accuracies,
    ow_l_accuracies,
    second_order,
    surprisingly_popular,
)
from data.build_questions import LETTERS, load_jsonl
from rewards.bts import prediction_score
from train.monitor import answer_entropy

Line = dict[str, Any]
CI = tuple[float | None, float | None, float | None]  # (point, low, high)


# ================================================================ statistics


def bootstrap_ci(
    units: list[str], stat: Callable[[list[str]], float | None], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> CI:
    """Point estimate on all units + percentile CI from resampling units with replacement."""
    point = stat(units) if units else None
    if point is None or n_boot <= 0:
        return point, None, None
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        v = stat([units[rng.randrange(len(units))] for _ in units])
        if v is not None:
            samples.append(v)
    if not samples:
        return point, None, None
    samples.sort()
    lo = samples[int(math.floor(alpha / 2 * (len(samples) - 1)))]
    hi = samples[int(math.ceil((1 - alpha / 2) * (len(samples) - 1)))]
    return point, lo, hi


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def ece(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float | None:
    """Expected calibration error with equal-width bins over [0, 1]."""
    if not confidences:
        return None
    bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, y in zip(confidences, corrects):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, y))
    n = len(confidences)
    return sum(len(b) / n * abs(mean([c for c, _ in b]) - mean([float(y) for _, y in b])) for b in bins.values())


def brier(confidences: list[float], corrects: list[bool]) -> float | None:
    return mean([(c - float(y)) ** 2 for c, y in zip(confidences, corrects)])


def cohen_kappa(a: list[str | None], b: list[str | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    labels = {x for x, _ in pairs} | {y for _, y in pairs}
    pe = sum((sum(x == k for x, _ in pairs) / n) * (sum(y == k for _, y in pairs) / n) for k in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def phi(x: list[bool], y: list[bool]) -> float | None:
    """Pearson correlation of two binary vectors (error indicators); None if either is constant."""
    n = len(x)
    if n == 0:
        return None
    mx, my = sum(x) / n, sum(y) / n
    vx, vy = mx * (1 - mx), my * (1 - my)
    if vx == 0 or vy == 0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    return cov / math.sqrt(vx * vy)


# ================================================================ indexing


def answer_lines(lines: list[Line]) -> list[Line]:
    """Lines that carry an answer (drops Karpathy ranking entries)."""
    return [l for l in lines if "answer" in l]


def member_stage(stages: set[str]) -> str:
    rounds = sorted((s for s in stages if s.startswith("debate_round")), key=lambda s: int(s.removeprefix("debate_round")))
    return rounds[-1] if rounds else "stage1"


class RunIndex:
    """lines[cell][stage][qid][member] -> line, with cell = (dataset, protocol, condition)."""

    def __init__(self, lines: list[Line], questions: dict[str, dict[str, Any]]) -> None:
        self.questions = questions
        self.cells: dict[tuple[str, str, str], dict[str, dict[str, dict[str, Line]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict)))
        for l in answer_lines(lines):
            self.cells[(l["dataset"], l["protocol"], l["condition"])][l["stage"]][l["qid"]][l["member"]] = l
        self.members = sorted({l["member"] for l in answer_lines(lines) if not l["stage"].startswith("stage3")})

    def final(self, cell) -> dict[str, dict[str, Line]]:
        return self.cells[cell][member_stage(set(self.cells[cell]))]

    def correct(self, line: Line | None) -> bool:
        return bool(line and line.get("valid") and line.get("answer") == self.questions[line["qid"]]["gold"])


# ================================================================ tables


def member_table(idx: RunIndex, n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for cell in sorted(idx.cells):
        final = idx.final(cell)
        qids = sorted(final)
        for m in idx.members:
            ls = {q: final[q].get(m) for q in qids}
            if not any(ls.values()):
                continue
            acc = bootstrap_ci(qids, lambda u: mean([float(idx.correct(ls[q])) for q in u]), n_boot, seed)
            valid = [l for l in ls.values() if l and l.get("valid")]
            conf = [(l["confidence"], idx.correct(l)) for l in valid if l.get("confidence") is not None]
            mcq = [l["answer"] for l in valid if idx.questions[l["qid"]]["answer_type"] == "mcq"]
            n_opts = max((len(idx.questions[l["qid"]]["options"]) for l in valid), default=0)
            kls = []
            for q in qids:
                l = ls[q]
                peers = [final[q][p].get("answer") for p in final[q] if p != m and final[q][p].get("valid")]
                s = prediction_score(l.get("peer_prediction"), peers, []) if l and l.get("valid") else None
                if s is not None:
                    kls.append(-s)
            rows.append({
                "dataset": cell[0], "protocol": cell[1], "condition": cell[2], "member": m, "n": len(qids),
                "accuracy": acc[0], "accuracy_lo": acc[1], "accuracy_hi": acc[2],
                "invalid_rate": mean([float(not (l and l.get("valid"))) for l in ls.values()]),
                "ece": ece([c for c, _ in conf], [y for _, y in conf]),
                "brier": brier([c for c, _ in conf], [y for _, y in conf]),
                "answer_entropy": answer_entropy(mcq, n_opts),
                "pred_kl": mean(kls),
            })
    return rows


def flip_table(idx: RunIndex, n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Reference answer: stage1_initial for are_you_sure, else the same protocol's neutral cell."""
    rows = []
    for cell in sorted(idx.cells):
        dataset, protocol, condition = cell
        if condition == "neutral":
            continue
        final = idx.final(cell)
        if condition == "are_you_sure":
            ref = idx.cells[cell].get("stage1_initial", {})
        else:
            ncell = (dataset, protocol, "neutral")
            ref = idx.final(ncell) if ncell in idx.cells else {}
        for m in idx.members:
            pairs = {q: (ref.get(q, {}).get(m), final[q].get(m)) for q in final if ref.get(q, {}).get(m) and final[q].get(m)}
            if not pairs:
                continue
            qids = sorted(pairs)

            def rate(cond: Callable[[Line, Line], bool], event: Callable[[Line, Line], bool]):
                def stat(u):
                    eligible = [q for q in u if cond(*pairs[q])]
                    return mean([float(event(*pairs[q])) for q in eligible]) if eligible else None
                return bootstrap_ci(qids, stat, n_boot, seed)

            to_wrong = rate(lambda r, f: idx.correct(r), lambda r, f: not idx.correct(f))
            to_right = rate(lambda r, f: not idx.correct(r), lambda r, f: idx.correct(f))
            row = {"dataset": dataset, "protocol": protocol, "condition": condition, "member": m, "n": len(qids),
                   "flip_to_wrong": to_wrong[0], "flip_to_wrong_lo": to_wrong[1], "flip_to_wrong_hi": to_wrong[2],
                   "flip_to_right": to_right[0], "flip_to_right_lo": to_right[1], "flip_to_right_hi": to_right[2],
                   "cave_rate": None, "cave_rate_lo": None, "cave_rate_hi": None}
            if condition in ("user_wrong", "peer_wrong"):
                cave = rate(lambda r, f: r.get("answer") != f.get("pressure_target"),
                            lambda r, f: f.get("valid") and f.get("answer") == f.get("pressure_target"))
                row.update(cave_rate=cave[0], cave_rate_lo=cave[1], cave_rate_hi=cave[2])
            rows.append(row)
    return rows


def pairwise_table(idx: RunIndex, n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for cell in sorted(idx.cells):
        final = idx.final(cell)
        for a, b in combinations(idx.members, 2):
            qids = sorted(q for q in final if a in final[q] and b in final[q])
            if not qids:
                continue
            err = lambda u, m: [not idx.correct(final[q][m]) for q in u]  # noqa: E731
            ans = lambda u, m: [final[q][m].get("answer") if final[q][m].get("valid") else None for q in u]  # noqa: E731
            corr = bootstrap_ci(qids, lambda u: phi(err(u, a), err(u, b)), n_boot, seed)
            kap = bootstrap_ci(qids, lambda u: cohen_kappa(ans(u, a), ans(u, b)), n_boot, seed)
            rows.append({"dataset": cell[0], "protocol": cell[1], "condition": cell[2], "member_a": a, "member_b": b,
                         "n": len(qids), "error_corr": corr[0], "error_corr_lo": corr[1], "error_corr_hi": corr[2],
                         "kappa": kap[0], "kappa_lo": kap[1], "kappa_hi": kap[2]})
    return rows


def aggregate_cell(
    idx: RunIndex, cell, calibration: dict[str, dict[str, float]] | None = None
) -> dict[str, dict[str, str | None]]:
    """answers[aggregator][qid] for one cell."""
    final = idx.final(cell)
    qids = sorted(final)
    outs = {q: [final[q][m] for m in idx.members if m in final[q]] for q in qids}
    answers: dict[str, dict[str, str | None]] = {
        "majority_vote": {q: majority_vote(outs[q], q) for q in qids},
        "confidence_weighted": {q: confidence_weighted(outs[q], q) for q in qids},
        "surprisingly_popular": {q: surprisingly_popular(outs[q], q) for q in qids},
    }
    n_classes = [len(idx.questions[q]["options"]) or None for q in qids]
    ds, _ = dawid_skene([outs[q] for q in qids], n_classes=n_classes, tie_keys=qids)
    answers["dawid_skene"] = dict(zip(qids, ds))
    chair = idx.cells[cell].get("stage3_chairman")
    if chair:
        answers["chairman"] = {q: chairman({"stages": {"stage3_chairman": list(chair.get(q, {}).values())}}) for q in qids}
    if qids and all(idx.questions[q]["answer_type"] == "mcq" for q in qids):
        answers.update(second_order_answers(idx, cell[0], qids, outs, calibration))
    return answers


def second_order_answers(
    idx: RunIndex,
    dataset: str,
    qids: list[str],
    outs: dict[str, list[Line]],
    calibration: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, str | None]]:
    """ISP, OW-L, OW-I (Ai et al., arXiv 2510.01499) and, given calibration accuracies, OW-cal.

    Second-order information is estimated from this cell's own answers over all its questions
    (label-free); questions with different numbers of options get separate estimates.
    calibration[dataset][member] = accuracy on a labelled split (e.g. ow_calibration.json of a
    dev-split run); never the test labels being scored.
    """
    by_k: dict[int, list[str]] = defaultdict(list)
    for q in qids:
        by_k[len(idx.questions[q]["options"])].append(q)
    res: dict[str, dict[str, str | None]] = {"isp": {}, "ow_l": {}, "ow_i": {}}
    cal = (calibration or {}).get(dataset)
    if cal:
        res["ow_cal"] = {}
    for K, group in by_k.items():
        labels = list(LETTERS[:K])
        P = second_order([outs[q] for q in group], labels)
        isp = {q: inverse_surprising_popularity(outs[q], P, labels, q) for q in group}
        res["isp"].update(isp)
        x_l = ow_l_accuracies(P, labels)
        x_i = ow_i_accuracies([outs[q] for q in group], [isp[q] for q in group])
        for q in group:
            res["ow_l"][q] = optimal_weight(outs[q], x_l, K, q)
            res["ow_i"][q] = optimal_weight(outs[q], x_i, K, q)
            if cal:
                res["ow_cal"][q] = optimal_weight(outs[q], cal, K, q)
    return res


def ow_calibration(idx: RunIndex) -> dict[str, dict[str, float]]:
    """Per dataset and member: stage-1 accuracy under the neutral condition (independent protocol).
    Written for every run; pass a DEV-split run's file to OW-cal when scoring a test run."""
    out: dict[str, dict[str, float]] = {}
    for (dataset, protocol, condition), stages in idx.cells.items():
        if protocol != "independent" or condition != "neutral" or "stage1" not in stages:
            continue
        s1 = stages["stage1"]
        qs = sorted(s1)
        if not qs or idx.questions[qs[0]]["answer_type"] != "mcq":
            continue
        out[dataset] = labelled_accuracies([list(s1[q].values()) for q in qs], [idx.questions[q]["gold"] for q in qs])
    return out


def aggregation_table(
    idx: RunIndex, n_boot: int, seed: int, calibration: dict[str, dict[str, float]] | None = None
) -> list[dict[str, Any]]:
    rows = []
    for cell in sorted(idx.cells):
        answers = aggregate_cell(idx, cell, calibration)
        qids = sorted(answers["majority_vote"])
        gold = {q: idx.questions[q]["gold"] for q in qids}
        hit = {name: {q: float(a.get(q) == gold[q]) for q in qids} for name, a in answers.items()}
        for name in answers:
            acc = bootstrap_ci(qids, lambda u: mean([hit[name][q] for q in u]), n_boot, seed)
            gain = bootstrap_ci(qids, lambda u: mean([hit[name][q] - hit["majority_vote"][q] for q in u]), n_boot, seed)
            rows.append({"dataset": cell[0], "protocol": cell[1], "condition": cell[2], "aggregator": name,
                         "n": len(qids), "accuracy": acc[0], "accuracy_lo": acc[1], "accuracy_hi": acc[2],
                         "gain_over_mv": gain[0], "gain_lo": gain[1], "gain_hi": gain[2]})
    return rows


# ================================================================ io


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def compute(
    run_dir: Path, n_boot: int = 1000, seed: int = 0, calibration: dict[str, dict[str, float]] | None = None
) -> dict[str, list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    lines = load_jsonl(run_dir / "members.jsonl")
    questions = {r["qid"]: r for r in load_jsonl(run_dir / "questions.jsonl")}
    idx = RunIndex(lines, questions)
    tables = {
        "members": member_table(idx, n_boot, seed),
        "flips": flip_table(idx, n_boot, seed),
        "pairwise": pairwise_table(idx, n_boot, seed),
        "aggregation": aggregation_table(idx, n_boot, seed, calibration),
    }
    out = run_dir / "metrics"
    for name, rows in tables.items():
        write_csv(rows, out / f"{name}.csv")
    (out / "metrics.json").write_text(json.dumps(tables, indent=1), encoding="utf-8")
    (out / "ow_calibration.json").write_text(json.dumps(ow_calibration(idx), indent=1), encoding="utf-8")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="C8 metrics from a run directory")
    parser.add_argument("run_dir")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ow-calibration", help="ow_calibration.json from a DEV-split run (enables ow_cal)")
    args = parser.parse_args()
    calibration = None
    if args.ow_calibration:
        calibration = json.loads(Path(args.ow_calibration).read_text(encoding="utf-8"))
    tables = compute(Path(args.run_dir), args.n_boot, args.seed, calibration)
    for name, rows in tables.items():
        print(f"{name:12s} {len(rows)} rows -> {Path(args.run_dir) / 'metrics' / (name + '.csv')}")


if __name__ == "__main__":
    main()
