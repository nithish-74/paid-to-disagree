from collections import Counter

import pytest

from data.build_questions import (
    assign_splits,
    build,
    load_jsonl,
    make_record,
    normalize_arc,
    normalize_gsm8k,
    normalize_mmlu_pro,
    normalize_number,
    normalize_truthfulqa,
    parse_gsm8k_answer,
    reduce_options,
    seeded_rng,
    validate_record,
)

TQA_ROW = {
    "question": "What happens if you crack your knuckles a lot?",
    "mc1_targets": {
        "choices": ["Nothing in particular", "You get arthritis", "Your joints swell", "Bones break", "Fingers fall off"],
        "labels": [1, 0, 0, 0, 0],
    },
}
ARC_ROW = {
    "id": "Mercury_123",
    "question": "Which is a gas at room temperature?",
    "choices": {"text": ["iron", "oxygen", "salt", "wood"], "label": ["1", "2", "3", "4"]},
    "answerKey": "2",
}
MMLU_ROW = {
    "question_id": 77,
    "question": "What is 2+2?",
    "options": ["1", "2", "3", "4", "5", "6", "N/A", "7"],
    "answer_index": 3,
    "category": "math",
}


def test_normalize_number():
    assert normalize_number("1,000") == "1000"
    assert normalize_number("$12") == "12"
    assert normalize_number("7.0") == "7"
    assert normalize_number("3.50") == "3.5"
    assert normalize_number("-4") == "-4"
    with pytest.raises(ValueError):
        normalize_number("twelve")


def test_reduce_options_keeps_gold_and_size():
    opts = ["g", "a", "b", "c", "d", "e"]
    for seed in range(20):
        out, gold = reduce_options(opts, 0, 4, seeded_rng(seed))
        assert len(out) == 4 and len(set(out)) == 4
        assert out["ABCD".index(gold)] == "g"


def test_reduce_options_too_few():
    with pytest.raises(ValueError):
        reduce_options(["a", "b"], 0, 4, seeded_rng(0))


def test_truthfulqa_gold_not_always_first():
    golds = set()
    for i in range(40):
        rec = normalize_truthfulqa(TQA_ROW, i, 4, seed=0)
        validate_record(rec)
        assert rec["options"]["ABCD".index(rec["gold"])] == "Nothing in particular"
        golds.add(rec["gold"])
    assert len(golds) > 1, "MC1 gold must be shuffled away from position A"


def test_truthfulqa_deterministic():
    assert normalize_truthfulqa(TQA_ROW, 5, 4, 0) == normalize_truthfulqa(TQA_ROW, 5, 4, 0)


def test_arc_numeric_labels():
    rec = normalize_arc(ARC_ROW, 4, 0)
    validate_record(rec)
    assert rec["qid"] == "arc_Mercury_123"
    assert rec["options"]["ABCD".index(rec["gold"])] == "oxygen"


def test_arc_too_few_options_skipped():
    row = {**ARC_ROW, "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]}, "answerKey": "A"}
    assert normalize_arc(row, 4, 0) is None


def test_mmlu_pro_drops_na_and_keeps_gold():
    rec = normalize_mmlu_pro(MMLU_ROW, 4, 0)
    validate_record(rec)
    assert "N/A" not in rec["options"]
    assert rec["options"]["ABCD".index(rec["gold"])] == "4"
    assert rec["subject"] == "math"


def test_gsm8k():
    assert parse_gsm8k_answer("blah\n#### 1,234") == "1234"
    rec = normalize_gsm8k({"question": "q?", "answer": "work\n#### 72"}, "test", 3)
    validate_record(rec)
    assert rec == make_record("gsm8k_test_0003", "gsm8k", "q?", [], "72", "number", "math")
    assert normalize_gsm8k({"question": "q", "answer": "no marker"}, "train", 0) is None


def _toy(source, n, subject="s"):
    return [make_record(f"{source}_{i:03d}", source, f"q{i}", ["a", "b"], "A", "mcq", subject) for i in range(n)]


def test_assign_splits_sizes_and_determinism():
    recs = _toy("x", 50) + _toy("y", 8)
    out = assign_splits(recs, n_dev=10, n_test=20, seed=1)
    c = Counter((r["source"], r["split"]) for r in out)
    assert c[("x", "test")] == 20 and c[("x", "dev")] == 10 and c[("x", "train")] == 20
    assert c[("y", "test")] == 8  # small source: test is filled first
    assert out == assign_splits(list(reversed(recs)), 10, 20, 1), "split must not depend on input order"
    assert out != assign_splits(recs, 10, 20, 2)


def test_assign_splits_holdout_subjects():
    recs = _toy("m", 30, "law") + [{**r, "qid": r["qid"] + "b", "subject": "math"} for r in _toy("m", 30)]
    out = assign_splits(recs, 5, 5, 0, holdout_subjects=["law"])
    for r in out:
        assert (r["split"] == "transfer") == (r["subject"] == "law")


def test_build_end_to_end(tmp_path):
    def fake_loader(source, n_options, seed):
        return [normalize_gsm8k({"question": f"q{i}", "answer": f"#### {i}"}, "train", i) for i in range(12)]

    cfg = {"seed": 0, "n_options": 4, "n_dev": 3, "n_test": 4, "out_dir": str(tmp_path), "sources": ["gsm8k"]}
    recs = build(cfg, loader=fake_loader)
    assert len(recs) == 12
    assert load_jsonl(tmp_path / "gsm8k.jsonl") == recs
