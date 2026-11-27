import json
from pathlib import Path

import pytest
import yaml

from analysis.combine_seeds import combine
from council.protocols import PROTOCOLS
from data.build_questions import save_jsonl
from eval.metrics import compute
from eval.run_eval import build_members, resolve_loras, run
from tests.fakes import FakeBackend
from tests.test_eval import QS, scripted

EVAL_CONFIGS = sorted(Path("configs").glob("eval*.yaml"))
REQUIRED = {"run_id", "seed", "council_config", "temperature", "datasets", "split", "conditions", "protocols"}


def load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", EVAL_CONFIGS, ids=[p.name for p in EVAL_CONFIGS])
def test_eval_configs_are_well_formed(path):
    cfg = load(path)
    assert REQUIRED <= set(cfg), REQUIRED - set(cfg)
    assert cfg["split"] in ("test", "transfer")
    assert Path(cfg["council_config"]).exists()
    assert {p["name"] for p in cfg["protocols"]} <= set(PROTOCOLS)
    assert set(cfg["conditions"]) <= {"neutral", "user_wrong", "user_right", "are_you_sure", "peer_wrong"}
    assert all(d.startswith("data/processed/") and d.endswith(".jsonl") for d in cfg["datasets"])


def test_main_eval_fits_one_commit():
    """The 12 h Kaggle commit runs 3 seeds of configs/eval.yaml: no B2 / 2-round debate there."""
    protos = [(p["name"], p.get("rounds", 1)) for p in load("configs/eval.yaml")["protocols"]]
    assert protos == [("independent", 1), ("karpathy_style", 1), ("debate", 1)]
    extra = {(p["name"], p.get("rounds")) for p in load("configs/eval_extra.yaml")["protocols"]}
    assert ("self_consistency", None) in extra and ("debate", 2) in extra


def test_b9_eval_pair_differs_only_in_the_lora():
    b9, base = load("configs/eval_b9.yaml"), load("configs/eval_b9_base.yaml")
    assert b9["loras_from_state"] == "runs/b9_single_qwen/state.json" and base["loras_from_state"] is None
    strip = lambda c: {k: v for k, v in c.items() if k not in ("run_id", "loras_from_state")}  # noqa: E731
    assert strip(b9) == strip(base)
    assert [m["name"] for m in load(b9["council_config"])["members"]] == ["qwen"]
    assert load("configs/train_b9.yaml")["experiment"] == "b9_single_qwen"  # state.json path matches


def test_b9_eval_builds_qwen_with_the_trained_lora(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"loras": {"qwen": "runs/b9_single_qwen/round1_qwen/stepfinal"}}), encoding="utf-8")
    cfg = {**load("configs/eval_b9.yaml"), "loras_from_state": str(state)}
    assert resolve_loras(cfg) == {"qwen": "runs/b9_single_qwen/round1_qwen/stepfinal"}
    seen = []
    members = build_members(cfg, backend_factory=lambda **kw: seen.append(kw) or FakeBackend(lambda c: "x"))
    assert [m.name for m in members] == ["qwen"]
    assert seen[0]["lora_path"].endswith("stepfinal") and seen[0]["gpu_memory_utilization"] == 0.85
    seen.clear()
    build_members(load("configs/eval_b9_base.yaml"), backend_factory=lambda **kw: seen.append(kw) or FakeBackend(lambda c: "x"))
    assert "lora_path" not in seen[0]


def test_one_member_council_runs_through_metrics_and_combine(tmp_path):
    data = tmp_path / "arc.jsonl"
    save_jsonl(QS, data)
    runs = []
    for seed in (0, 1):
        cfg = {**load("configs/eval_b9_base.yaml"), "run_id": f"s{seed}", "seed": seed, "datasets": [str(data)]}
        out = tmp_path / f"s{seed}"
        run(cfg, out, [scripted("qwen", "B", caves=True)])
        compute(out, n_boot=20)
        runs.append(out)
    counts = combine(runs, tmp_path / "combined")
    assert counts["pairwise"] == 0 and counts["members"] > 0 and counts["flips"] > 0
    with (tmp_path / "combined" / "flips.csv").open(encoding="utf-8") as f:
        import csv
        uw = [r for r in csv.DictReader(f) if r["condition"] == "user_wrong"]
    assert uw and float(uw[0]["cave_rate_mean"]) == 1.0
