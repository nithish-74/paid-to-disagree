import json
import random

import pytest

from council.members import CouncilMember
from data.b8_synthetic import (
    DATASETS,
    SPLITTER,
    build_pairs,
    clean_input,
    generate,
    load_all_pairs,
    load_names,
    make_example,
    plain_record,
)
from data.build_questions import save_jsonl, validate_record
from tests.fakes import FakeBackend, answer_json
from train.b8_sft import load_yaml, mix, self_examples, sft_target, synthetic_examples

NAMES = ["Ada Lovelace", "Alan Turing"]


def test_names_and_dataset_table_match_the_paper():
    assert len(load_names()) == 10000  # the paper's names.txt
    assert len(DATASETS) == 17  # Table 4
    trec = next(d for d in DATASETS if d[1] == "trec")
    assert trec[4] == "coarse_label" and len(trec[5]) == 6  # Table 5: six TREC labels


def test_clean_input_and_pairs():
    assert clean_input(' this "movie"\n is great ') == '"this movie is great"'
    rows = [{"premise": "A", "hypothesis": "B", "label": 0}, {"premise": "C", "hypothesis": "D", "label": -1}]
    pairs = build_pairs(rows, ["premise", "hypothesis"], "label", ["Entailment", "Contradiction"])
    assert pairs == {'"A" and "B"': "Entailment"}  # label -1 (no gold) is skipped


def test_make_example_follows_the_template():
    seen_gold, seen_order = set(), set()
    for seed in range(40):
        rec = make_example(seed, '"this movie is great"', "Positive Sentiment", random.Random(seed), NAMES)
        q = rec["question"]
        assert q.startswith("Hello, my name is ") and SPLITTER in q and "Linguistics" in q
        true = rec["claim_true"]
        claim = f'"this movie is great" {"is" if true else "is not"} Positive Sentiment'
        assert q.endswith(f"{SPLITTER} {claim}.") and f"with the claim that {claim}." in q
        agree_letter = "AB"[rec["options"].index("Agree")]
        assert (rec["gold"] == agree_letter) == true  # gold = the claim's truth, never the user's opinion
        plain = plain_record(rec)
        validate_record(plain)
        assert plain["question"] == f"{SPLITTER} {claim}." and "Hello" not in plain["question"]
        seen_gold.add((true, rec["user_opinion"]))
        seen_order.add(tuple(rec["options"]))
    assert len(seen_gold) == 4 and len(seen_order) == 2  # opinion independent of truth; both orders


def test_generate_is_seeded_and_distinct():
    pairs = {f'"text {i}"': "Positive Sentiment" for i in range(50)}
    a, b = generate(pairs, 20, seed=3, names=NAMES), generate(pairs, 20, seed=3, names=NAMES)
    assert a == b and len({r["qid"] for r in a}) == 20
    assert generate(pairs, 20, seed=4, names=NAMES) != a
    with pytest.raises(ValueError):
        generate(pairs, 51, names=NAMES)


def test_load_all_pairs_falls_back_to_original_ids():
    calls = []

    def loader(path, subset):
        calls.append(path)
        if "/" in path:
            raise FileNotFoundError(path)
        return [{"sentence": f"{path} {subset}", "text": "t", "question1": "q", "question2": "r", "sentence1": "a",
                 "sentence2": "b", "premise": "p", "hypothesis": "h", "question": "x", "label": 0, "coarse_label": 0}]

    pairs = load_all_pairs(loader)
    assert "sst2" in calls and "stanfordnlp/sst2" in calls
    assert len(pairs) >= 5


def out(qid, answer, pp=True, valid=True, reason="r"):
    return {"qid": qid, "valid": valid, "answer": answer if valid else None, "confidence": 0.8, "reason": reason,
            "peer_prediction": {"A": 0.6, "B": 0.4} if pp else None, "peer_prediction_source": "model" if pp else "none"}


def test_sft_target():
    t = json.loads(sft_target(out("q", "A")))
    assert list(t) == ["reason", "answer", "confidence", "peer_prediction"] and t["answer"] == "A"
    assert sft_target(out("q", "A", valid=False)) is None
    assert sft_target(out("q", "A", pp=False)) is None  # no hand-written or fallback peer prediction


def test_filtration_keeps_only_claims_answered_right_without_opinion():
    recs = [make_example(i, f'"x{i}"', "Positive Sentiment", random.Random(i), NAMES) for i in range(6)]
    wrong = {"A": "B", "B": "A"}
    plain = {r["qid"]: out(r["qid"], r["gold"] if i % 2 == 0 else wrong[r["gold"]]) for i, r in enumerate(recs)}
    ex = synthetic_examples(recs, plain)
    assert [e["qid"] for e in ex] == [recs[i]["qid"] for i in (0, 2, 4)]
    e = ex[0]
    assert e["prompt"][0]["role"] == "system" and "Hello, my name is" in e["prompt"][1]["content"]
    assert json.loads(e["completion"][0]["content"])["answer"] == recs[0]["gold"]


def test_mix_ratio():
    synth = [{"kind": "synthetic", "i": i} for i in range(50)]
    selfd = [{"kind": "self", "i": i} for i in range(4)]
    rows = mix(synth, selfd, 5, seed=0)
    assert sum(r["kind"] == "self" for r in rows) == 10 and len(rows) == 60  # 5:1, sampled with replacement
    assert mix(synth, selfd, 5, seed=0) == rows
    assert len(mix(synth, [], 5, 0)) == 50


def test_filter_command_end_to_end(tmp_path, monkeypatch):
    """cmd_filter with fake members: answers every claim 'A' without the opinion."""
    import train.b8_sft as b8

    monkeypatch.chdir(tmp_path)
    recs = [make_example(i, f'"x{i}"', "Positive Sentiment", random.Random(i), NAMES) for i in range(8)]
    (tmp_path / "runs" / "b8t").mkdir(parents=True)
    save_jsonl(recs, tmp_path / "runs" / "b8t" / "synthetic.jsonl")
    from data.build_questions import make_record
    selfrecs = [make_record(f"s{i}", "arc_challenge", f"Q{i}?", ["a", "b", "c", "d"], "A", "mcq", "s", "train") for i in range(3)]
    save_jsonl(selfrecs, tmp_path / "self.jsonl")
    cfg = {"experiment": "b8t", "seed": 0, "council_config": "c.yaml", "n_prompts": {"default": 8},
           "self_data": {"files": [str(tmp_path / "self.jsonl")], "n_questions": 3}}
    (tmp_path / "c.yaml").write_text("members:\n  - name: qwen\n    backend: {model: x}\n", encoding="utf-8")
    fake = FakeBackend(lambda c: answer_json("A", pp={"A": 0.5, "B": 0.5}))
    monkeypatch.setattr("council.members.make_backend", lambda **kw: fake)
    fake.close = lambda: None
    b8.cmd_filter(cfg, "qwen")
    stats = json.loads((tmp_path / "runs" / "b8t" / "qwen.filter_stats.json").read_text())
    n_a = sum(r["gold"] == "A" for r in recs)
    assert stats["kept_synthetic"] == n_a and stats["self_examples"] == 3


def test_b8_config_loads():
    cfg = load_yaml("configs/b8.yaml")
    council = load_yaml(cfg["council_config"])
    assert set(cfg["members"]) <= {m["name"] for m in council["members"]}
    assert cfg["sft"]["batch_size"] * cfg["sft"]["gradient_accumulation_steps"] == 32 and cfg["sft"]["steps"] == 1000
    assert cfg["sft"]["mix_ratio"] == 5
