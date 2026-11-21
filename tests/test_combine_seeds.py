import csv
import json

import pytest

from analysis.combine_seeds import combine, combine_rows


def write(run, seed, rows):
    (run / "metrics").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"seed": seed}), encoding="utf-8")
    for table in ("members", "flips", "pairwise", "aggregation"):
        with (run / "metrics" / f"{table}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def row(acc, lo, hi, ece=""):
    return {"dataset": "arc", "protocol": "independent", "condition": "neutral", "member": "qwen", "n": "50",
            "accuracy": str(acc), "accuracy_lo": str(lo), "accuracy_hi": str(hi), "ece": str(ece)}


def test_combine_rows_mean_sd_and_intervals():
    [r] = combine_rows([[row(0.5, 0.4, 0.6, 0.1)], [row(0.6, 0.5, 0.7, "")], [row(0.7, 0.55, 0.8, 0.3)]])
    assert r["n_seeds"] == 3 and r["n_per_seed"] == 50
    assert r["accuracy_mean"] == pytest.approx(0.6) and r["accuracy_sd"] == pytest.approx(0.1)
    assert r["accuracy_seed_lo"] == pytest.approx(0.6 - 4.303 * 0.1 / 3 ** 0.5)
    assert (r["accuracy_boot_lo"], r["accuracy_boot_hi"]) == (0.4, 0.8)
    assert r["ece_mean"] == pytest.approx(0.2)  # empty value skipped
    assert "ece_boot_lo" not in r  # no CI columns for ece


def test_combine_writes_tables_and_rejects_duplicate_seeds(tmp_path):
    runs = []
    for s, acc in enumerate((0.5, 0.6, 0.7)):
        runs.append(tmp_path / f"s{s}")
        write(runs[-1], s, [row(acc, acc - 0.1, acc + 0.1)])
    counts = combine(runs, tmp_path / "out")
    assert counts == {"members": 1, "flips": 1, "pairwise": 1, "aggregation": 1}
    with (tmp_path / "out" / "members.csv").open(encoding="utf-8") as f:
        assert float(next(csv.DictReader(f))["accuracy_mean"]) == pytest.approx(0.6)
    dup = tmp_path / "dup"
    write(dup, 0, [row(0.5, 0.4, 0.6)])
    with pytest.raises(ValueError):
        combine([runs[0], dup], tmp_path / "out2")
