import csv
import json
import math

import pytest

from council.members import CouncilMember
from council.protocols import RANK_SYSTEM
from data.build_questions import make_record, save_jsonl
from data.pressure import ARE_YOU_SURE
from eval.metrics import bootstrap_ci, brier, cohen_kappa, compute, ece, member_stage, phi
from eval.run_eval import build_members, resolve_loras, run
from tests.fakes import FakeBackend, answer_json, last_user

# ---------------------------------------------------------------- statistics with known answers


def test_ece_and_brier():
    assert ece([1.0, 1.0], [True, True]) == 0.0
    assert ece([0.9] * 10, [True] * 6 + [False] * 4) == pytest.approx(0.3)
    assert brier([1.0, 0.0], [True, True]) == pytest.approx(0.5)
    assert ece([], []) is None


def test_cohen_kappa():
    assert cohen_kappa(list("ABAB"), list("ABAB")) == 1.0
    # po = 0.5, pe = 0.5 -> kappa 0
    assert cohen_kappa(list("AABB"), list("ABAB")) == pytest.approx(0.0)
    assert cohen_kappa(list("AB"), list("BA")) == pytest.approx(-1.0)
    assert cohen_kappa([None], ["A"]) is None


def test_phi():
    assert phi([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert phi([True, False, True, False], [False, True, False, True]) == pytest.approx(-1.0)
    assert phi([True, True], [True, False]) is None  # constant vector


def test_bootstrap_ci_brackets_point_and_is_seeded():
    units = [str(i) for i in range(100)]
    vals = {u: float(int(u) % 2) for u in units}
    stat = lambda u: sum(vals[x] for x in u) / len(u)  # noqa: E731
    p, lo, hi = bootstrap_ci(units, stat, n_boot=300, seed=1)
    assert p == 0.5 and lo < 0.5 < hi and 0.3 < lo and hi < 0.7
    assert bootstrap_ci(units, stat, 300, 1) == (p, lo, hi)
    assert bootstrap_ci([], stat) == (None, None, None)


def test_member_stage():
    assert member_stage({"stage1", "stage1_initial"}) == "stage1"
    assert member_stage({"stage1", "debate_round1", "debate_round2"}) == "debate_round2"


# ---------------------------------------------------------------- end to end with scripted members

QS = [make_record(f"q{i}", "arc_challenge", f"Q{i}?", ["w", "x", "y", "z"], "B", "mcq", "s", "test") for i in range(8)]
PP = {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}


def scripted(name, neutral, caves=False, unsure_flip=False):
    """neutral: answer under no pressure. caves: adopts the pushed answer. unsure_flip: changes on 'Are you sure?'."""

    def reply(conv):
        if conv[0]["content"] == RANK_SYSTEM:
            return json.dumps({"ranking": [1, 2, 3]})
        msg = last_user(conv)
        if msg == ARE_YOU_SURE:
            return answer_json("D" if unsure_flip else neutral, 0.9, PP)
        if caves and "fairly sure the answer is" in msg:
            pushed = msg.split("fairly sure the answer is ")[1][0]
            return answer_json(pushed, 0.9, PP)
        return answer_json(neutral, 0.9, PP)

    return CouncilMember(name, FakeBackend(reply))


@pytest.fixture
def run_dir(tmp_path):
    data = tmp_path / "arc.jsonl"
    save_jsonl(QS, data)
    cfg = {"run_id": "t", "seed": 0, "datasets": [str(data)], "split": "test", "n": 0,
           "conditions": ["neutral", "user_wrong", "user_right", "are_you_sure", "peer_wrong"],
           "protocols": [{"name": "independent"}, {"name": "karpathy_style"}, {"name": "debate", "rounds": 1}]}
    members = [scripted("honest", "B"), scripted("sycophant", "B", caves=True), scripted("wrong", "C", unsure_flip=True)]
    out = tmp_path / "run"
    n = run(cfg, out, members)
    assert n > 0
    return out


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(rows, **kw):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


def test_run_eval_writes_lines_and_questions(run_dir):
    lines = [json.loads(l) for l in (run_dir / "members.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {l["protocol"] for l in lines} == {"independent", "karpathy_style", "debate"}
    assert {l["stage"] for l in lines} >= {"stage1", "stage1_initial", "stage2_rank", "stage3_chairman", "debate_round1"}
    assert all("gold" not in l for l in lines)
    uw = [l for l in lines if l["condition"] == "user_wrong" and "answer" in l]
    assert all(l["pressure_target"] not in (None, "B") for l in uw)
    assert len((run_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_metrics_end_to_end(run_dir):
    tables = compute(run_dir, n_boot=50)
    members = tables["members"]
    neutral = {r["member"]: r for r in pick(members, protocol="independent", condition="neutral")}
    assert neutral["honest"]["accuracy"] == 1.0 and neutral["wrong"]["accuracy"] == 0.0
    assert neutral["honest"]["invalid_rate"] == 0.0
    # confidence 0.9 everywhere: honest ECE 0.1, wrong ECE 0.9
    assert neutral["honest"]["ece"] == pytest.approx(0.1) and neutral["wrong"]["ece"] == pytest.approx(0.9)

    flips = tables["flips"]
    syc = pick(flips, protocol="independent", condition="user_wrong", member="sycophant")[0]
    hon = pick(flips, protocol="independent", condition="user_wrong", member="honest")[0]
    assert syc["cave_rate"] == 1.0 and syc["flip_to_wrong"] == 1.0
    assert hon["cave_rate"] == 0.0 and hon["flip_to_wrong"] == 0.0
    ur = pick(flips, protocol="independent", condition="user_right", member="wrong")[0]
    assert ur["flip_to_right"] == 0.0  # 'wrong' never accepts the user's correct answer
    ays = pick(flips, protocol="independent", condition="are_you_sure", member="wrong")[0]
    assert ays["flip_to_wrong"] is None and ays["n"] == 8  # never right initially -> no eligible questions
    peer = pick(flips, protocol="independent", condition="peer_wrong", member="honest")[0]
    assert peer["cave_rate"] == 0.0

    agg = {r["aggregator"]: r for r in pick(tables["aggregation"], protocol="independent", condition="neutral")}
    assert agg["majority_vote"]["accuracy"] == 1.0 and agg["majority_vote"]["gain_over_mv"] == 0.0
    assert set(agg) == {"majority_vote", "confidence_weighted", "surprisingly_popular", "dawid_skene",
                        "isp", "ow_l", "ow_i"}
    assert agg["isp"]["n"] == agg["majority_vote"]["n"]
    kp = {r["aggregator"] for r in pick(tables["aggregation"], protocol="karpathy_style", condition="neutral")}
    assert "chairman" in kp

    # ow_calibration.json from this run, fed back in, adds the labelled-calibration OW variant
    cal = json.loads((run_dir / "metrics" / "ow_calibration.json").read_text(encoding="utf-8"))
    assert set(cal["arc_challenge"]) == {"honest", "sycophant", "wrong"}
    with_cal = compute(run_dir, n_boot=50, seed=0, calibration=cal)
    agg_cal = {r["aggregator"] for r in pick(with_cal["aggregation"], protocol="independent", condition="neutral")}
    assert "ow_cal" in agg_cal

    # honest & sycophant both always right under neutral -> error vectors constant -> phi undefined
    pw = pick(tables["pairwise"], protocol="independent", condition="neutral", member_a="honest", member_b="sycophant")[0]
    assert pw["error_corr"] is None and pw["kappa"] == 1.0

    assert (run_dir / "metrics" / "aggregation.csv").exists()
    assert read_csv(run_dir / "metrics" / "members.csv")[0]["dataset"] == "arc_challenge"


def test_resolve_loras_and_build_members(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"loras": {"qwen": "L1", "llama": "L2"}}), encoding="utf-8")
    assert resolve_loras({"loras": {"llama": "X"}, "loras_from_state": str(state)}) == {"qwen": "L1", "llama": "X"}
    council = tmp_path / "council.yaml"
    council.write_text("members:\n  - {name: qwen, backend: {model: m}}\n", encoding="utf-8")
    seen = {}
    members = build_members({"council_config": str(council), "seed": 2, "loras_from_state": str(state)},
                            backend_factory=lambda **kw: seen.update(kw) or FakeBackend(lambda c: "x"))
    assert members[0].seed == 2 and seen["lora_path"] == "L1" and math.isclose(seen["gpu_memory_utilization"], 0.28)


def test_build_members_per_member_gpu_share(tmp_path):
    council = tmp_path / "council.yaml"
    council.write_text("members:\n  - {name: qwen, backend: {model: m}}\n  - {name: gemma, backend: {model: g}}\n",
                       encoding="utf-8")
    seen = []
    build_members({"council_config": str(council), "seed": 0, "gpu_memory_utilization": {"gemma": 0.38}},
                  backend_factory=lambda **kw: seen.append(kw) or FakeBackend(lambda c: "x"))
    assert [kw["gpu_memory_utilization"] for kw in seen] == [0.28, 0.38]


# ---------------------------------------------------------------- B2 self-consistency + 2-round debate


class SeedBackend(FakeBackend):
    """Sampling stand-in: the answer depends on the seed (B on seeds 0 and 1 mod 3, else C)."""

    def __init__(self):
        super().__init__(lambda c: "")
        self.calls = []

    def chat(self, conversations, temperature, max_tokens, seed=None):
        self.calls.append((temperature, seed))
        if conversations and conversations[0][0]["content"] == RANK_SYSTEM:
            return [json.dumps({"ranking": [1, 2, 3]})] * len(conversations)
        letter = "B" if (seed or 0) % 3 in (0, 1) else "C"
        return [answer_json(letter, 0.9, PP)] * len(conversations)


def test_self_consistency_samples_per_member():
    from council.protocols import self_consistency
    from data.pressure import apply_condition

    backend = SeedBackend()
    member = CouncilMember("solo", backend)
    prompts = [apply_condition(q, "neutral") for q in QS[:2]] + [apply_condition(QS[0], "are_you_sure")]
    res = self_consistency(prompts, {q["qid"]: q for q in QS}, [member], samples=3, temperature=0.7, seed=0)
    assert len(res) == 3 and {r.protocol for r in res} == {"self_consistency_solo"}
    assert [o["member"] for o in res[0].stages["stage1"]] == ["solo#s0", "solo#s1", "solo#s2"]
    assert [o["answer"] for o in res[0].stages["stage1"]] == ["B", "B", "C"]
    assert "stage1_initial" in res[2].stages and "stage1_initial" not in res[0].stages
    assert {t for t, _ in backend.calls} == {0.7}
    assert member.stats.n == 0 and member.temperature == 0.0  # original member untouched
    with pytest.raises(ValueError):
        self_consistency(prompts, {}, [member], temperature=0.0)


def test_eval_with_b2_and_two_round_debate(tmp_path):
    data = tmp_path / "arc.jsonl"
    save_jsonl(QS, data)
    cfg = {"run_id": "t", "seed": 0, "datasets": [str(data)], "split": "test", "n": 0,
           "conditions": ["neutral", "user_wrong"],
           "protocols": [{"name": "independent"}, {"name": "debate", "rounds": 1}, {"name": "debate", "rounds": 2},
                         {"name": "self_consistency", "samples": 3, "temperature": 0.7}]}
    members = [scripted("honest", "B"), scripted("sycophant", "B", caves=True),
               CouncilMember("sampler", SeedBackend())]
    out = tmp_path / "run"
    run(cfg, out, members)
    lines = [json.loads(l) for l in (out / "members.jsonl").read_text(encoding="utf-8").splitlines()]
    protocols = {l["protocol"] for l in lines}
    assert {"debate", "debate2", "self_consistency_honest", "self_consistency_sampler"} <= protocols
    assert {l["stage"] for l in lines if l["protocol"] == "debate2"} >= {"debate_round1", "debate_round2"}

    tables = compute(out, n_boot=20)
    b2 = {r["aggregator"]: r for r in pick(tables["aggregation"], protocol="self_consistency_sampler", condition="neutral")}
    assert b2["majority_vote"]["accuracy"] == 1.0  # samples B, B, C -> B
    council = pick(tables["members"], protocol="independent", condition="neutral")
    assert {r["member"] for r in council} == {"honest", "sycophant", "sampler"}  # no "#s" samples in council cells
