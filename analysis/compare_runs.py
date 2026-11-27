"""Compare a trained setup with the untrained baseline, cell by cell (hypotheses H1, H3, H4, H5).

Both inputs are analysis.combine_seeds output directories (several seeds each). For every row key
(dataset, protocol, condition, member / pair / aggregator) present in both, and every metric a
hypothesis uses, the table gives:

    base_mean, trained_mean, diff = trained - base,
    diff_lo / diff_hi  95% Welch t-interval from the two sets of per-seed estimates
                       (se = sqrt(sd_b^2/n_b + sd_t^2/n_t), Welch-Satterthwaite df)
    expected           the direction the hypothesis predicts (-1 = lower is better, +1 = higher)
    supported          the interval excludes 0 on the expected side (needs >= 2 seeds on each side)

Hypotheses (architecture doc):
    H1  flip_to_wrong and cave_rate drop under user_wrong / are_you_sure
    H5  cave_rate drops under peer_wrong
    H3  pairwise error_corr drops (neutral)
    H4  gain_over_mv of ISP / OW / surprisingly popular grows (neutral)
    guard  neutral accuracy should not drop (reported, not a hypothesis)
Member-level rows use the independent protocol; aggregation uses every protocol.

    python -m analysis.compare_runs runs/eval_untrained_combined runs/eval_cross_model_v1_combined \\
        --out runs/compare_cross_model_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from analysis.combine_seeds import KEY_COLUMNS, T95, as_float, read_table

H4_AGGREGATORS = {"isp", "ow_l", "ow_i", "ow_cal", "surprisingly_popular"}


def welch(mb: float, sb: float | None, nb: int, mt: float, st: float | None, nt: int) -> tuple[float | None, float | None]:
    """95% interval for mt - mb from per-seed means/sds; None when either side has < 2 seeds."""
    if sb is None or st is None or nb < 2 or nt < 2:
        return None, None
    vb, vt = sb * sb / nb, st * st / nt
    se = math.sqrt(vb + vt)
    diff = mt - mb
    if se == 0:
        return diff, diff
    df = (vb + vt) ** 2 / ((vb * vb) / (nb - 1) + (vt * vt) / (nt - 1)) if (vb or vt) else nb + nt - 2
    t = T95.get(max(1, int(df)), 1.96)
    return diff - t * se, diff + t * se


def hypothesis(table: str, row: dict[str, str], metric: str) -> tuple[str, int] | None:
    """(hypothesis, expected sign) a cell/metric tests, or None if it is not part of the comparison."""
    cond, proto = row.get("condition"), row.get("protocol")
    if table == "flips" and proto == "independent":
        if metric in ("flip_to_wrong", "cave_rate") and cond in ("user_wrong", "are_you_sure"):
            return "H1", -1
        if metric == "cave_rate" and cond == "peer_wrong":
            return "H5", -1
    if table == "pairwise" and proto == "independent" and cond == "neutral" and metric == "error_corr":
        return "H3", -1
    if table == "aggregation" and cond == "neutral" and metric == "gain_over_mv" and row.get("aggregator") in H4_AGGREGATORS:
        return "H4", +1
    if table == "members" and proto == "independent" and cond == "neutral" and metric == "accuracy":
        return "guard", +1
    return None


METRICS = {"members": ["accuracy"], "flips": ["flip_to_wrong", "cave_rate"], "pairwise": ["error_corr"],
           "aggregation": ["gain_over_mv"]}


def compare(base_dir: Path, trained_dir: Path) -> list[dict[str, Any]]:
    out = []
    for table, metrics in METRICS.items():
        bp, tp = Path(base_dir) / f"{table}.csv", Path(trained_dir) / f"{table}.csv"
        if not bp.exists() or not tp.exists():
            continue
        base_rows = read_table(bp)
        keys = [c for c in (base_rows[0] if base_rows else {}) if c in KEY_COLUMNS]
        base = {tuple(r[k] for k in keys): r for r in base_rows}
        for tr in read_table(tp):
            k = tuple(tr.get(c, "") for c in keys)
            br = base.get(k)
            if br is None:
                continue
            for m in metrics:
                h = hypothesis(table, tr, m)
                if h is None:
                    continue
                mb, mt = as_float(br.get(f"{m}_mean", "")), as_float(tr.get(f"{m}_mean", ""))
                if mb is None or mt is None:
                    continue
                nb, nt = int(as_float(br.get("n_seeds", "")) or 0), int(as_float(tr.get("n_seeds", "")) or 0)
                lo, hi = welch(mb, as_float(br.get(f"{m}_sd", "")), nb, mt, as_float(tr.get(f"{m}_sd", "")), nt)
                sign = h[1]
                supported = lo is not None and ((sign < 0 and hi < 0) or (sign > 0 and lo > 0))
                out.append({"hypothesis": h[0], "table": table, **dict(zip(keys, k)), "metric": m,
                            "base_mean": mb, "trained_mean": mt, "diff": mt - mb, "diff_lo": lo, "diff_hi": hi,
                            "expected": sign, "supported": supported, "n_seeds_base": nb, "n_seeds_trained": nt})
    return out


def summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per hypothesis: cells tested, cells moving the expected way, cells with interval support,
    and cells with support for the OPPOSITE direction (reported, never hidden)."""
    s: dict[str, dict[str, int]] = {}
    for r in rows:
        d = s.setdefault(r["hypothesis"], {"cells": 0, "expected_direction": 0, "supported": 0, "opposite_supported": 0})
        d["cells"] += 1
        d["expected_direction"] += int(r["diff"] * r["expected"] > 0)
        d["supported"] += int(r["supported"])
        lo, hi = r["diff_lo"], r["diff_hi"]
        d["opposite_supported"] += int(lo is not None and ((r["expected"] < 0 and lo > 0) or (r["expected"] > 0 and hi < 0)))
    return s


def write(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [f for f in r if f not in fields]
    with (out_dir / "compare.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    (out_dir / "summary.json").write_text(json.dumps(summary(rows), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trained vs untrained comparison (H1, H3, H4, H5)")
    parser.add_argument("base", type=Path, help="combine_seeds output of the untrained baseline")
    parser.add_argument("trained", type=Path, help="combine_seeds output of the trained setup")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.base, args.trained)
    write(rows, args.out)
    print(json.dumps(summary(rows), indent=2))
    print(f"{len(rows)} cells -> {args.out / 'compare.csv'}")


if __name__ == "__main__":
    main()
