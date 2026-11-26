import json

import pytest

from analysis.combine_seeds import combine
from analysis.paper_tables import build, fmt, supported_nonzero
from data.build_questions import save_jsonl
from eval.metrics import compute
from eval.run_eval import run
from tests.test_eval import QS, scripted


def s(mean, slo, shi, blo, bhi):
    return {"mean": mean, "seed_lo": slo, "seed_hi": shi, "boot_lo": blo, "boot_hi": bhi, "n_seeds": 3}


def test_supported_needs_both_intervals_to_exclude_zero():
    assert supported_nonzero(s(0.1, 0.05, 0.15, 0.02, 0.2))
    assert supported_nonzero(s(-0.1, -0.15, -0.05, -0.2, -0.01))
    assert not supported_nonzero(s(0.1, 0.05, 0.15, -0.01, 0.2))   # bootstrap interval crosses 0
    assert not supported_nonzero(s(0.1, -0.02, 0.2, 0.01, 0.2))    # seed interval crosses 0
    assert not supported_nonzero(s(0.1, None, None, 0.01, 0.2))


def test_fmt():
    assert fmt(s(0.4213, 0.35, 0.49, 0.3, 0.5)) == "42.1 [35.0, 49.0]"
    assert fmt(s(0.05, 0.02, 0.08, 0.01, 0.09), signed=True, mark=True) == "+5.0 [2.0, 8.0] †"
    assert fmt(s(None, None, None, None, None)) == "–"


@pytest.fixture
def combined(tmp_path):
    data = tmp_path / "arc.jsonl"
    save_jsonl(QS, data)
    runs = []
    for seed in (0, 1, 2):
        cfg = {"run_id": f"t{seed}", "seed": seed, "datasets": [str(data)], "split": "test", "n": 0,
               "conditions": ["neutral", "user_wrong", "user_right", "are_you_sure", "peer_wrong"],
               "protocols": [{"name": "independent"}, {"name": "karpathy_style"}, {"name": "debate", "rounds": 1}]}
        members = [scripted("honest", "B"), scripted("sycophant", "B", caves=True),
                   scripted("wrong", "C", unsure_flip=True)]
        out = tmp_path / f"seed{seed}"
        run(cfg, out, members)
        compute(out, n_boot=30)
        runs.append(out)
    combine(runs, tmp_path / "combined")
    return tmp_path / "combined"


def test_build_tables_and_figures(combined, tmp_path):
    out = tmp_path / "paper"
    written = build(combined, out)
    md = (out / "tables.md").read_text(encoding="utf-8")
    assert "Table 1. Caving" in md and "Table 3a" in md and "Appendix A" in md
    assert "\\begin{tabular}" in (out / "tables.tex").read_text(encoding="utf-8")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    cave = summary["cave_rate"]["user_wrong"]["arc_challenge"]
    assert cave["sycophant"] == 1.0 and cave["honest"] == 0.0           # scripted behaviour shows up
    assert summary["aggregation"]["neutral"]["independent/majority_vote"]["arc_challenge"]["accuracy"] == 1.0
    assert "karpathy_style/chairman" in summary["aggregation"]["neutral"]
    pytest.importorskip("matplotlib")
    assert set(written["figures"]) == {"fig_error_corr.pdf", "fig_error_corr.png", "fig_flips.pdf", "fig_flips.png"}
    assert (out / "fig_flips.png").stat().st_size > 1000


def test_build_tolerates_missing_tables(tmp_path):
    (tmp_path / "combined").mkdir()
    written = build(tmp_path / "combined", tmp_path / "paper")
    assert (tmp_path / "paper" / "tables.md").exists() and written["figures"] == []
