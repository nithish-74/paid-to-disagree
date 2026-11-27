import csv
import json

import pytest

from analysis.compare_runs import compare, hypothesis, summary, welch, write


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def flips_row(cond, member, cave, sd, n=3):
    return {"dataset": "arc_challenge", "protocol": "independent", "condition": cond, "member": member,
            "n_seeds": n, "cave_rate_mean": cave, "cave_rate_sd": sd, "flip_to_wrong_mean": cave, "flip_to_wrong_sd": sd}


def test_welch_interval():
    lo, hi = welch(0.5, 0.02, 3, 0.3, 0.02, 3)
    assert lo < -0.2 < hi < 0  # clear drop, interval below zero
    assert welch(0.5, None, 1, 0.3, 0.02, 3) == (None, None)  # one seed: no interval
    lo, hi = welch(0.5, 0.2, 3, 0.45, 0.2, 3)
    assert lo < 0 < hi  # noisy: interval spans zero


def test_hypothesis_mapping():
    assert hypothesis("flips", {"condition": "user_wrong", "protocol": "independent"}, "cave_rate") == ("H1", -1)
    assert hypothesis("flips", {"condition": "peer_wrong", "protocol": "independent"}, "cave_rate") == ("H5", -1)
    assert hypothesis("flips", {"condition": "user_right", "protocol": "independent"}, "cave_rate") is None
    assert hypothesis("pairwise", {"condition": "neutral", "protocol": "independent"}, "error_corr") == ("H3", -1)
    assert hypothesis("aggregation", {"condition": "neutral", "protocol": "debate", "aggregator": "isp"}, "gain_over_mv") == ("H4", 1)
    assert hypothesis("aggregation", {"condition": "neutral", "protocol": "debate", "aggregator": "majority_vote"}, "gain_over_mv") is None


def test_compare_end_to_end(tmp_path):
    base, trained = tmp_path / "base", tmp_path / "trained"
    write_csv(base / "flips.csv", [flips_row("user_wrong", "qwen", 0.60, 0.02), flips_row("peer_wrong", "qwen", 0.40, 0.02)])
    write_csv(trained / "flips.csv", [flips_row("user_wrong", "qwen", 0.30, 0.02), flips_row("peer_wrong", "qwen", 0.55, 0.02)])
    pw = {"dataset": "arc_challenge", "protocol": "independent", "condition": "neutral", "member_a": "llama",
          "member_b": "qwen", "n_seeds": 3, "error_corr_mean": 0.5, "error_corr_sd": 0.3}
    write_csv(base / "pairwise.csv", [pw])
    write_csv(trained / "pairwise.csv", [{**pw, "error_corr_mean": 0.45}])
    rows = compare(base, trained)
    by = {(r["hypothesis"], r["metric"], r.get("condition")): r for r in rows}
    h1 = by[("H1", "cave_rate", "user_wrong")]
    assert h1["diff"] == pytest.approx(-0.30) and h1["supported"]
    h5 = by[("H5", "cave_rate", "peer_wrong")]
    assert h5["diff"] > 0 and not h5["supported"]  # worse: reported, not supported
    assert not by[("H3", "error_corr", "neutral")]["supported"]  # small, noisy drop
    s = summary(rows)
    assert s["H1"] == {"cells": 2, "expected_direction": 2, "supported": 2, "opposite_supported": 0}
    assert s["H5"]["opposite_supported"] == 1
    write(rows, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["H3"]["cells"] == 1
    assert (tmp_path / "out" / "compare.csv").read_text().startswith("hypothesis,")
