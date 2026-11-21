"""Combine eval.metrics tables from several seeds into the numbers we report.

For every row key (dataset, protocol, condition, member / pair / aggregator) and every
metric, the combined table gives:
    <metric>_mean        mean of the per-seed point estimates
    <metric>_sd          standard deviation across seeds
    <metric>_seed_lo/hi  95% t-interval across seeds (seed-to-seed variation)
    <metric>_boot_lo/hi  widest per-seed 95% bootstrap interval (question resampling)
plus n_seeds. A claim needs both kinds of interval to agree. Rows present in only some
seeds are kept, with n_seeds showing how many contributed.

    python -m analysis.combine_seeds runs/eval_untrained_seed0 runs/eval_untrained_seed1 runs/eval_untrained_seed2 \
        --out runs/eval_untrained_combined
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

TABLES = ("members", "flips", "pairwise", "aggregation")
KEY_COLUMNS = ("dataset", "protocol", "condition", "member", "member_a", "member_b", "aggregator")
# two-sided 95% Student t critical values by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def as_float(v: str) -> float | None:
    if v in ("", "None", None):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_columns(header: list[str]) -> list[str]:
    """Point-estimate columns: everything numeric that is not a key, n, or a CI bound."""
    return [c for c in header if c not in KEY_COLUMNS and c != "n" and not c.endswith(("_lo", "_hi"))]


def combine_rows(per_seed: list[list[dict[str, str]]]) -> list[dict[str, object]]:
    header = list(per_seed[0][0].keys()) if per_seed and per_seed[0] else []
    keys = [c for c in header if c in KEY_COLUMNS]
    metrics = metric_columns(header)
    grouped: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    order: list[tuple] = []
    for rows in per_seed:
        for row in rows:
            k = tuple(row[c] for c in keys)
            if k not in grouped:
                order.append(k)
            grouped[k].append(row)

    out = []
    for k in order:
        rows = grouped[k]
        rec: dict[str, object] = dict(zip(keys, k))
        rec["n_seeds"] = len(rows)
        ns = [as_float(r.get("n", "")) for r in rows]
        rec["n_per_seed"] = int(statistics.mean(x for x in ns if x is not None)) if any(x is not None for x in ns) else None
        for m in metrics:
            vals = [x for x in (as_float(r[m]) for r in rows) if x is not None]
            mean = statistics.mean(vals) if vals else None
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            rec[f"{m}_mean"] = mean
            rec[f"{m}_sd"] = sd
            if sd is not None:
                half = T95.get(len(vals) - 1, 1.96) * sd / math.sqrt(len(vals))
                rec[f"{m}_seed_lo"], rec[f"{m}_seed_hi"] = mean - half, mean + half
            else:
                rec[f"{m}_seed_lo"] = rec[f"{m}_seed_hi"] = None
            if f"{m}_lo" in header:
                los = [x for x in (as_float(r[f"{m}_lo"]) for r in rows) if x is not None]
                his = [x for x in (as_float(r[f"{m}_hi"]) for r in rows) if x is not None]
                rec[f"{m}_boot_lo"] = min(los) if los else None
                rec[f"{m}_boot_hi"] = max(his) if his else None
        out.append(rec)
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        fields += [f for f in r if f not in fields]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def combine(run_dirs: list[Path], out_dir: Path) -> dict[str, int]:
    if len(run_dirs) < 3:
        print(f"warning: {len(run_dirs)} seed(s); the protocol asks for at least 3 before claiming anything")
    seeds = []
    for d in run_dirs:
        cfg = d / "config.json"
        seeds.append(json.loads(cfg.read_text(encoding="utf-8")).get("seed") if cfg.exists() else None)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"run dirs share a seed: {dict(zip(map(str, run_dirs), seeds))}")
    counts = {}
    for table in TABLES:
        per_seed = [read_table(d / "metrics" / f"{table}.csv") for d in run_dirs if (d / "metrics" / f"{table}.csv").exists()]
        if len(per_seed) != len(run_dirs):
            raise FileNotFoundError(f"{table}.csv missing in some run dirs; run eval.metrics on each first")
        rows = combine_rows(per_seed)
        write_csv(rows, out_dir / f"{table}.csv")
        counts[table] = len(rows)
    (out_dir / "sources.json").write_text(
        json.dumps({"runs": [str(d) for d in run_dirs], "seeds": seeds}, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-seed metrics tables")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for table, n in combine(args.runs, args.out).items():
        print(f"{table:12s} {n} rows -> {args.out / (table + '.csv')}")


if __name__ == "__main__":
    main()
