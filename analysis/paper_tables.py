"""Paper tables and figures from the seed-combined metrics (analysis/combine_seeds.py output).

Reads runs/<eval>_combined/{members,flips,pairwise,aggregation}.csv and writes to --out:
    tables.md          every table in Markdown (for reading / the appendix)
    tables.tex         the main tables in LaTeX (booktabs)
    fig_error_corr.pdf/.png   pairwise error correlation heatmaps (H3)
    fig_flips.pdf/.png        flip-to-wrong vs flip-to-right per condition
    summary.json       headline numbers, so the text can quote them without copying by hand

Numbers are mean over seeds with the 95% interval across seeds. A difference is marked as
supported (†) only when BOTH the seed interval and the widest per-seed bootstrap interval
exclude zero, the rule stated in combine_seeds.py.

    python -m analysis.paper_tables runs/eval_untrained_combined --out analysis/paper_untrained
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PRESSURE = ("user_wrong", "peer_wrong", "user_right", "are_you_sure")
CONDITION_NAMES = {
    "neutral": "neutral",
    "user_wrong": "user pushes wrong",
    "peer_wrong": "peer argues wrong",
    "user_right": "user pushes right",
    "are_you_sure": "\"are you sure?\"",
}
AGGREGATORS = ("majority_vote", "confidence_weighted", "surprisingly_popular", "dawid_skene", "chairman")


# ---------------------------------------------------------------- reading


def num(v: Any) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read(combined: Path, table: str) -> list[dict[str, str]]:
    path = combined / f"{table}.csv"
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(rows: list[dict[str, str]], **kw: str) -> list[dict[str, str]]:
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def stat(row: dict[str, str] | None, metric: str) -> dict[str, float | None]:
    if row is None:
        return {"mean": None, "seed_lo": None, "seed_hi": None, "boot_lo": None, "boot_hi": None, "n_seeds": None}
    return {"mean": num(row.get(f"{metric}_mean")), "seed_lo": num(row.get(f"{metric}_seed_lo")),
            "seed_hi": num(row.get(f"{metric}_seed_hi")), "boot_lo": num(row.get(f"{metric}_boot_lo")),
            "boot_hi": num(row.get(f"{metric}_boot_hi")), "n_seeds": num(row.get("n_seeds"))}


def supported_nonzero(s: dict[str, float | None]) -> bool:
    """Both intervals exclude zero (same side)."""
    lo1, hi1, lo2, hi2 = s["seed_lo"], s["seed_hi"], s["boot_lo"], s["boot_hi"]
    if None in (lo1, hi1, lo2, hi2):
        return False
    return (lo1 > 0 and lo2 > 0) or (hi1 < 0 and hi2 < 0)


def fmt(s: dict[str, float | None], pct: bool = True, signed: bool = False, mark: bool = False) -> str:
    m = s["mean"]
    if m is None:
        return "–"
    k = 100 if pct else 1
    d = 1 if pct else 2
    sign = "+" if signed and m > 0 else ""
    text = f"{sign}{m * k:.{d}f}"
    if s["seed_lo"] is not None and s["seed_hi"] is not None:
        text += f" [{s['seed_lo'] * k:.{d}f}, {s['seed_hi'] * k:.{d}f}]"
    if mark and supported_nonzero(s):
        text += " †"
    return text


# ---------------------------------------------------------------- tables


def caving_table(flips: list[dict[str, str]], protocol: str = "independent") -> dict[str, Any]:
    """Table 1: cave rate (moved to the pushed wrong answer) per dataset x member."""
    members = sorted({r["member"] for r in flips})
    datasets = sorted({r["dataset"] for r in flips})
    out = {"columns": members, "blocks": {}}
    for cond in ("user_wrong", "peer_wrong"):
        block = {}
        for ds in datasets:
            block[ds] = {m: stat(next(iter(pick(flips, dataset=ds, protocol=protocol, condition=cond, member=m)), None),
                                 "cave_rate") for m in members}
        out["blocks"][cond] = block
    return out


def flips_table(flips: list[dict[str, str]], protocol: str = "independent") -> dict[str, Any]:
    """Table 2: flip-to-wrong / flip-to-right per condition x member, macro-averaged over datasets.
    Macro averages have no interval; per-dataset values with intervals are in the appendix."""
    members = sorted({r["member"] for r in flips})
    out: dict[str, Any] = {"columns": members, "rows": {}, "per_dataset": {}}
    for cond in PRESSURE:
        rows = pick(flips, protocol=protocol, condition=cond)
        if not rows:
            continue
        for metric in ("flip_to_wrong", "flip_to_right"):
            vals = {}
            for m in members:
                xs = [num(r.get(f"{metric}_mean")) for r in rows if r["member"] == m]
                xs = [x for x in xs if x is not None]
                vals[m] = statistics.mean(xs) if xs else None
            out["rows"][(cond, metric)] = vals
            out["per_dataset"][(cond, metric)] = {
                (r["dataset"], r["member"]): stat(r, metric) for r in rows}
    return out


def aggregation_table(agg: list[dict[str, str]], condition: str = "neutral") -> dict[str, Any]:
    """Table 3: accuracy of each protocol x aggregator, and gain over majority vote."""
    datasets = sorted({r["dataset"] for r in agg})
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for r in pick(agg, condition=condition):
        cells[(r["protocol"], r["aggregator"])] = cells.get((r["protocol"], r["aggregator"]), {})
        cells[(r["protocol"], r["aggregator"])][r["dataset"]] = {
            "accuracy": stat(r, "accuracy"), "gain": stat(r, "gain_over_mv")}
    order = sorted(cells, key=lambda k: (k[0] != "independent", k[0],
                                         AGGREGATORS.index(k[1]) if k[1] in AGGREGATORS else 99, k[1]))
    return {"datasets": datasets, "rows": [(k, cells[k]) for k in order], "condition": condition}


def members_table(members: list[dict[str, str]]) -> dict[str, Any]:
    """Appendix: each member alone (neutral, independent): accuracy, invalid rate, ECE, prediction KL."""
    rows = pick(members, protocol="independent", condition="neutral")
    return {"rows": sorted(((r["dataset"], r["member"]), {
        "accuracy": stat(r, "accuracy"), "invalid_rate": num(r.get("invalid_rate_mean")),
        "ece": num(r.get("ece_mean")), "brier": num(r.get("brier_mean")),
        "pred_kl": num(r.get("pred_kl_mean")), "answer_entropy": num(r.get("answer_entropy_mean")),
    }) for r in rows)}


def error_corr(pairwise: list[dict[str, str]], protocol: str = "independent", condition: str = "neutral"):
    """dataset -> {(a, b): mean error correlation}."""
    out: dict[str, dict[tuple[str, str], float | None]] = defaultdict(dict)
    for r in pick(pairwise, protocol=protocol, condition=condition):
        out[r["dataset"]][(r["member_a"], r["member_b"])] = num(r.get("error_corr_mean"))
    return dict(out)


# ---------------------------------------------------------------- rendering


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def tex_table(caption: str, label: str, header: list[str], rows: list[list[str]]) -> str:
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%").replace("†", r"$^\dagger$")  # noqa: E731
    cols = "l" + "c" * (len(header) - 1)
    body = "\n".join(" & ".join(esc(c) for c in r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[t]\n\\centering\\small\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
            f"{' & '.join(esc(h) for h in header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{esc(caption)}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def render(combined: Path) -> tuple[str, str, dict[str, Any]]:
    flips, agg = read(combined, "flips"), read(combined, "aggregation")
    members, pairwise = read(combined, "members"), read(combined, "pairwise")
    md, tex, summary = ["# Paper tables", f"Source: `{combined}`. Values: % mean over seeds [95% interval across "
                        "seeds]; † = supported (seed and bootstrap intervals both exclude 0)."], [], {}

    cav = caving_table(flips)
    rows = []
    for cond, block in cav["blocks"].items():
        for ds, vals in block.items():
            rows.append([f"{CONDITION_NAMES[cond]} / {ds}"] + [fmt(vals[m]) for m in cav["columns"]])
            summary.setdefault("cave_rate", {}).setdefault(cond, {})[ds] = {m: vals[m]["mean"] for m in cav["columns"]}
    header = ["condition / dataset"] + cav["columns"]
    md += ["## Table 1. Caving: share of answers that move to the pushed wrong answer", md_table(header, rows)]
    tex.append(tex_table("Caving rate (%) of untrained members under user and peer pressure (independent "
                         "protocol), mean over seeds with 95% interval across seeds.", "tab:caving", header, rows))

    fl = flips_table(flips)
    rows = [[f"{CONDITION_NAMES[c]}: {m.replace('_', '-')}"] +
            [("–" if v[mm] is None else f"{v[mm] * 100:.1f}") for mm in fl["columns"]]
            for (c, m), v in fl["rows"].items()]
    header = ["condition: metric"] + fl["columns"]
    md += ["## Table 2. Harmful vs beneficial revision (macro-average over datasets, %)", md_table(header, rows)]
    tex.append(tex_table("Flip-to-wrong (harmful) and flip-to-right (beneficial) rates (%), macro-averaged over "
                         "datasets.", "tab:flips", header, rows))

    for cond in ("neutral", "user_wrong"):
        at = aggregation_table(agg, cond)
        if not at["rows"]:
            continue
        header = ["protocol / aggregator"] + at["datasets"]
        rows = []
        for (proto, name), by_ds in at["rows"]:
            cells = []
            for ds in at["datasets"]:
                c = by_ds.get(ds)
                if c is None:
                    cells.append("–")
                elif name == "majority_vote":
                    cells.append(fmt(c["accuracy"]))
                else:
                    cells.append(f"{fmt(c['accuracy'])}; {fmt(c['gain'], signed=True, mark=True)}")
                if c is not None:
                    summary.setdefault("aggregation", {}).setdefault(cond, {}).setdefault(f"{proto}/{name}", {})[ds] = {
                        "accuracy": c["accuracy"]["mean"], "gain_over_mv": c["gain"]["mean"],
                        "supported": supported_nonzero(c["gain"])}
            rows.append([f"{proto} / {name}"] + cells)
        md += [f"## Table 3{'a' if cond == 'neutral' else 'b'}. Accuracy and gain over majority vote "
               f"({CONDITION_NAMES[cond]})", "Cells: accuracy %; gain over MV in points.", md_table(header, rows)]
        if cond == "neutral":
            tex.append(tex_table("Accuracy (%) per protocol and aggregator, neutral condition; for non-MV "
                                 "aggregators also the gain over majority vote (points). † = supported.",
                                 "tab:aggregation", header, rows))

    corr = error_corr(pairwise)
    if corr:
        pairs = sorted({p for d in corr.values() for p in d})
        header = ["dataset"] + [f"{a}–{b}" for a, b in pairs]
        rows = [[ds] + [("–" if corr[ds].get(p) is None else f"{corr[ds][p]:.2f}") for p in pairs] for ds in sorted(corr)]
        md += ["## Table 4. Pairwise error correlation (phi), neutral, independent", md_table(header, rows)]
        summary["error_corr"] = {ds: {f"{a}-{b}": v for (a, b), v in d.items()} for ds, d in corr.items()}

    mt = members_table(members)
    if mt["rows"]:
        header = ["dataset / member", "accuracy %", "invalid %", "ECE", "Brier", "pred KL", "entropy"]
        f2 = lambda x, k=1, d=2: "–" if x is None else f"{x * k:.{d}f}"  # noqa: E731
        rows = [[f"{ds} / {m}", fmt(v["accuracy"]), f2(v["invalid_rate"], 100, 1), f2(v["ece"]), f2(v["brier"]),
                 f2(v["pred_kl"]), f2(v["answer_entropy"])] for (ds, m), v in mt["rows"]]
        md += ["## Appendix A. Members alone (neutral)", md_table(header, rows)]

    per_ds = fl["per_dataset"]
    if per_ds:
        rows = [[CONDITION_NAMES[c], metric.replace("_", "-"), ds, m, fmt(s)]
                for (c, metric), d in per_ds.items() for (ds, m), s in sorted(d.items())]
        md += ["## Appendix B. Flip rates per dataset (%)",
               md_table(["condition", "metric", "dataset", "member", "value"], rows)]
    return "\n\n".join(md) + "\n", "\n".join(tex), summary


# ---------------------------------------------------------------- figures


def figures(combined: Path, out: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: figures skipped (pip install matplotlib)")
        return []
    written = []
    corr = error_corr(read(combined, "pairwise"))
    if corr:
        datasets = sorted(corr)
        members = sorted({m for d in corr.values() for p in d for m in p})
        fig, axes = plt.subplots(1, len(datasets), figsize=(2.6 * len(datasets), 2.6), squeeze=False)
        for ax, ds in zip(axes[0], datasets):
            # diagonal (a model with itself) left blank; undefined pairs (a member never wrong) are grey too
            grid = [[None if a == b else (corr[ds].get((a, b)) if (a, b) in corr[ds] else corr[ds].get((b, a)))
                     for b in members] for a in members]
            grid = [[float("nan") if v is None else v for v in row] for row in grid]
            cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#e6e6e6")
            im = ax.imshow(grid, vmin=-1, vmax=1, cmap=cmap)
            ax.set_xticks(range(len(members)), members, rotation=45, fontsize=7)
            ax.set_yticks(range(len(members)), members, fontsize=7)
            ax.set_title(ds, fontsize=8)
            for i, row in enumerate(grid):
                for j, v in enumerate(row):
                    if v == v and i != j:
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="error correlation (phi)")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_error_corr.{ext}", bbox_inches="tight", dpi=200)
            written.append(f"fig_error_corr.{ext}")
        plt.close(fig)

    fl = flips_table(read(combined, "flips"))
    if fl["rows"]:
        conds = [c for c in PRESSURE if (c, "flip_to_wrong") in fl["rows"]]
        members = fl["columns"]
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
        width = 0.8 / max(len(members), 1)
        for ax, metric in zip(axes, ("flip_to_wrong", "flip_to_right")):
            for k, m in enumerate(members):
                ys = [(fl["rows"].get((c, metric), {}).get(m) or 0.0) * 100 for c in conds]
                ax.bar([i + k * width for i in range(len(conds))], ys, width, label=m)
            ax.set_xticks([i + width * (len(members) - 1) / 2 for i in range(len(conds))],
                          [CONDITION_NAMES[c] for c in conds], rotation=20, fontsize=7)
            ax.set_title(metric.replace("_", "-") + " (%)", fontsize=8)
        axes[0].legend(fontsize=7)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"fig_flips.{ext}", bbox_inches="tight", dpi=200)
            written.append(f"fig_flips.{ext}")
        plt.close(fig)
    return written


def build(combined: Path, out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    md, tex, summary = render(combined)
    (out / "tables.md").write_text(md, encoding="utf-8")
    (out / "tables.tex").write_text(tex, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return {"tables": ["tables.md", "tables.tex", "summary.json"], "figures": figures(combined, out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper tables and figures from combined metrics")
    parser.add_argument("combined", type=Path, help="output dir of analysis.combine_seeds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    written = build(args.combined, args.out)
    print(f"wrote {written['tables'] + written['figures']} -> {args.out}")


if __name__ == "__main__":
    main()
