import json
import random

import pytest

from council.members import CouncilMember
from data.build_questions import make_record
from data.pressure import ARE_YOU_SURE
from rewards.ca import ca_batch
from rewards.combined import RewardConfig
from tests.fakes import FakeBackend, answer_json
from train.grpo_data import build_rows
from train.grpo_roundrobin import (
    dev_prompts,
    grpo_args,
    load_records,
    load_state,
    next_leg,
    save_state,
    stale_cache_members,
)
from train.monitor import DevMonitor, EarlyStopper, answer_entropy, dev_metrics, flip_rate
from train.peer_cache import PeerCache, generate_samples
from train.reward_fn import completion_text, make_reward_fn, strip_gold

PP = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def rec(i, gold="B", split="train"):
    return make_record(f"q{i}", "arc_challenge", f"Question {i}?", ["w", "x", "y", "z"], gold, "mcq", "s", split)


RECS = [rec(i) for i in range(6)]


def sample(answer, valid=True, raw=None):
    return {"answer": answer if valid else None, "peer_prediction": PP, "confidence": 0.7, "valid": valid,
            "raw": raw or answer_json(answer or "A", 0.7, PP)}


def make_cache():
    cache = PeerCache()
    for r in RECS:
        cache.entries[("qwen", r["qid"])] = [sample("B"), sample("B"), sample("C")]
        cache.entries[("llama", r["qid"])] = [sample("B")]
        cache.entries[("gemma", r["qid"])] = [sample("C"), sample("A"), sample("A")]
    return cache


# ---------------------------------------------------------------- CA pool


def test_ca_pool_works_with_single_task():
    pool = [(f"p{i}", "ABCD"[i % 4], ["ABCD"[i % 4]]) for i in range(40)]
    scores = ca_batch([["A", "Z"]], [["A"]], random.Random(0), pool=pool, qids=["q"])
    assert scores[0][0] > scores[0][1]
    with pytest.raises(ValueError):
        ca_batch([["A"]], [["A"]], random.Random(0), pool=pool)


def test_ca_pool_constant_answers_earn_zero():
    pool = [(f"p{i}", "A", ["A", "A"]) for i in range(10)]
    assert ca_batch([["A"] * 3], [["A", "A"]], random.Random(0), pool=pool, qids=["q"]) == [[0.0, 0.0, 0.0]]


# ---------------------------------------------------------------- cache


def test_cache_majority_report_and_peers():
    cache = make_cache()
    assert cache.majority("qwen", "q0") == "B" and cache.majority("gemma", "q0") == "A"
    assert cache.peer_names("qwen") == ["gemma", "llama"]
    assert [p.answer for p in cache.peers_for("qwen", "q0")] == ["A", "B"]
    assert cache.peers_for("qwen", "missing")[0].usable is False
    pool = cache.pool("qwen")
    assert len(pool) == 6 and pool[0] == ("q0", "B", ["A", "B"])


def test_cache_save_load_and_update(tmp_path):
    cache = make_cache()
    cache.save(tmp_path / "c.jsonl")
    loaded = PeerCache.load(tmp_path / "c.jsonl")
    assert loaded.entries == cache.entries
    loaded.update_member("qwen", {"q0": [sample("D")]})
    assert loaded.majority("qwen", "q0") == "D" and loaded.samples("qwen", "q1") == []
    assert loaded.majority("llama", "q1") == "B"


def test_generate_samples_uses_n_seeds():
    seeds = []

    class SeedBackend(FakeBackend):
        def chat(self, conversations, temperature, max_tokens, seed=None):
            seeds.append(seed)
            return super().chat(conversations, temperature, max_tokens, seed)

    m = CouncilMember("qwen", SeedBackend(lambda c: answer_json("B", 0.7, PP)), temperature=0.7)
    out = generate_samples(m, RECS[:2], n_samples=3, seed=10)
    assert seeds == [10, 11, 12]
    assert all(len(v) == 3 and v[0]["answer"] == "B" for v in out.values())
    assert "gold" not in out["q0"][0]


# ---------------------------------------------------------------- rows


def test_build_rows_conditions_and_no_gold():
    rows = build_rows(RECS[:2], "qwen", make_cache())
    assert len(rows) == 2 * 4
    assert all(r["prompt"][0]["role"] == "system" for r in rows)
    assert all("gold" not in r and "B" != r.get("gold") for r in rows)
    ays = next(r for r in rows if r["condition"] == "are_you_sure")
    assert [m["role"] for m in ays["prompt"]] == ["system", "user", "assistant", "user"]
    assert ays["prompt"][-1]["content"] == ARE_YOU_SURE
    assert next(r for r in rows if r["condition"] == "neutral")["pressure_target"] == ""


def test_build_rows_skips_are_you_sure_without_cache_and_rejects_dev():
    rows = build_rows(RECS[:1], "qwen", PeerCache())
    assert {r["condition"] for r in rows} == {"neutral", "user_wrong", "user_right"}
    with pytest.raises(ValueError):
        build_rows([rec(9, split="dev")], "qwen", PeerCache())


# ---------------------------------------------------------------- reward fn


def _trl_call(fn, rows, completions_per_row):
    """Mimic TRL: each prompt repeated G times, columns passed as kwargs."""
    prompts, completions, cols = [], [], {"row_id": [], "qid": [], "condition": [], "pressure_target": []}
    for row, comps in zip(rows, completions_per_row):
        for c in comps:
            prompts.append(row["prompt"])
            completions.append([{"role": "assistant", "content": c}])
            for k in cols:
                cols[k].append(row[k])
    return fn(prompts, completions, **cols)


def test_reward_fn_groups_scores_and_logs(tmp_path):
    cache = make_cache()
    rows = [r for r in build_rows(RECS[:2], "qwen", cache) if r["condition"] == "neutral"]
    fn = make_reward_fn(cache, "qwen", {r["qid"]: r for r in RECS}, RewardConfig(), log_path=tmp_path / "t.jsonl")
    good, bad = answer_json("B", 0.8, PP), "not json"
    rewards = _trl_call(fn, rows, [[good, good, bad], [good, bad, bad]])
    assert len(rewards) == 6
    assert rewards[2] == RewardConfig().invalid_penalty and rewards[0] > rewards[2]
    log = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert log["n"] == 6 and log["invalid"] == pytest.approx(3 / 6)


def test_reward_fn_flip_under_user_wrong():
    cache = make_cache()
    [row] = [r for r in build_rows(RECS[:1], "qwen", cache) if r["condition"] == "user_wrong"]
    fn = make_reward_fn(cache, "qwen", {r["qid"]: r for r in RECS}, RewardConfig())
    held = answer_json("B", 0.8, PP)
    caved = answer_json(row["pressure_target"], 0.8, PP)
    r_held, r_caved = _trl_call(fn, [row], [[held, caved]])
    assert r_held > r_caved


def test_reward_fn_never_sees_gold():
    assert all("gold" not in r for r in strip_gold({r["qid"]: r for r in RECS}).values())
    with pytest.raises(ValueError):
        make_reward_fn(make_cache(), "qwen", {}, RewardConfig(), golds={"q0": "B"})
    fn = make_reward_fn(make_cache(), "qwen", {r["qid"]: r for r in RECS}, RewardConfig(oracle=True), golds={"q0": "B"})
    [row] = [r for r in build_rows(RECS[:1], "qwen", make_cache()) if r["condition"] == "neutral"]
    right, wrong = _trl_call(fn, [row], [[answer_json("B", 0.8, PP), answer_json("D", 0.8, PP)]])
    assert right > wrong


def test_completion_text():
    assert completion_text("x") == "x"
    assert completion_text([{"role": "assistant", "content": "y"}]) == "y"
    assert completion_text([]) == ""


# ---------------------------------------------------------------- monitor


def test_answer_entropy():
    assert answer_entropy(list("ABCD") * 5, 4) == pytest.approx(1.0)
    assert answer_entropy(["B"] * 10, 4) == 0.0
    assert answer_entropy([None], 4) is None


def test_flip_rate():
    neutral = {"a": "B", "b": "B", "c": "C"}
    pressured = {"a": ("C", "C"), "b": ("B", "C"), "c": ("C", "C")}  # c excluded: already believed C
    assert flip_rate(neutral, pressured) == 0.5
    assert flip_rate({}, {}) is None


def test_dev_metrics():
    recs = {r["qid"]: r for r in [rec(0, "A"), rec(1, "B")]}
    outs = [
        {"qid": "q0", "condition": "neutral", "answer": "A", "valid": True},
        {"qid": "q1", "condition": "neutral", "answer": "C", "valid": True},
        {"qid": "q0", "condition": "user_wrong", "answer": "D", "valid": True},
        {"qid": "q1", "condition": "user_wrong", "answer": None, "valid": False},
    ]
    m = dev_metrics(outs, recs, {"q0": "D", "q1": "A"})
    assert m["accuracy"] == 0.5 and m["flip_rate"] == 0.5 and m["invalid_rate"] == 0.25


def test_early_stopper():
    s = EarlyStopper(max_acc_drop=0.03, min_entropy=0.5)
    assert s.update(0, {"accuracy": 0.60, "entropy": 0.95}) == (False, "baseline")
    assert s.update(200, {"accuracy": 0.58, "entropy": 0.9})[0] is False and s.last_good_step == 200
    stop, reason = s.update(400, {"accuracy": 0.56, "entropy": 0.9})
    assert stop and "accuracy" in reason and s.last_good_step == 200
    s2 = EarlyStopper()
    s2.update(0, {"accuracy": 0.6, "entropy": 0.9})
    assert s2.update(200, {"accuracy": 0.61, "entropy": 0.2})[1].startswith("answer entropy")


def test_dev_monitor_every_n_steps(tmp_path):
    saved, metrics = [], iter([{"accuracy": 0.6, "entropy": 0.9}, {"accuracy": 0.6, "entropy": 0.9},
                               {"accuracy": 0.5, "entropy": 0.9}])
    mon = DevMonitor(every=200, save=lambda s: saved.append(s) or f"ck{s}", evaluate=lambda p: next(metrics),
                     stopper=EarlyStopper(), log_path=tmp_path / "m.jsonl")
    mon.start("ck0")
    assert not any(mon.on_step(s) for s in range(1, 200))
    assert mon.on_step(200) is False
    assert mon.on_step(400) is True and "accuracy" in mon.stop_reason
    assert saved == [200, 400] and mon.stopper.last_good_step == 200
    assert len((tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()) == 3


# ---------------------------------------------------------------- orchestration helpers


def test_next_leg_and_stale_cache(tmp_path):
    rot = ["qwen", "llama", "gemma"]
    state = load_state(tmp_path / "none.json")
    assert next_leg(state, rot, 2) == (1, "qwen") and stale_cache_members(state, rot) == []
    state["legs"].append({"round": 1, "member": "qwen", "lora": "L1"})
    state["loras"]["qwen"] = "L1"
    assert next_leg(state, rot, 2) == (1, "llama")
    assert stale_cache_members(state, rot) == ["qwen"]
    state["cache"]["qwen"] = "L1"
    assert stale_cache_members(state, rot) == []
    state["legs"] += [{"round": r, "member": m} for r in (1, 2) for m in rot]
    assert next_leg(state, rot, 2) is None
    save_state(state, tmp_path / "s.json")
    assert load_state(tmp_path / "s.json") == state


def test_grpo_args_and_validation(tmp_path):
    from train.grpo_roundrobin import load_yaml

    cfg = load_yaml("configs/train.yaml")
    args = grpo_args(cfg, tmp_path)
    assert args["fp16"] and not args["bf16"] and args["beta"] == 0.0 and args["num_generations"] == 8
    assert args["output_dir"] == str(tmp_path / "trainer")
    cfg["grpo"]["per_device_train_batch_size"] = 5  # 5 x 4 = 20, not a multiple of G = 8
    with pytest.raises(ValueError):
        grpo_args(cfg, tmp_path)


def test_load_records_and_dev_prompts(tmp_path):
    from data.build_questions import save_jsonl

    save_jsonl([rec(i, split="train") for i in range(5)] + [rec(9, split="dev")], tmp_path / "d.jsonl")
    tr = load_records([str(tmp_path / "d.jsonl")], "train", 3, seed=0)
    assert len(tr) == 3 and all(r["split"] == "train" for r in tr)
    assert tr == load_records([str(tmp_path / "d.jsonl")], "train", 3, seed=0)
    prompts, targets = dev_prompts([rec(9, split="dev")], seed=0)
    assert [p.condition for p in prompts] == ["neutral", "user_wrong"] and targets["q9"] != "B"
