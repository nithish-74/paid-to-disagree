import json

import pytest

from council.members import (
    RETRY_MESSAGE,
    CouncilMember,
    extract_json_object,
    load_members,
    parse_answer,
    parse_confidence,
    parse_distribution,
    parse_letter,
    run_in_batches,
    system_prompt,
    write_outputs,
)
from data.build_questions import make_record
from data.pressure import ARE_YOU_SURE, apply_condition
from tests.fakes import FakeBackend, answer_json, last_user

MCQ = make_record("arc_1", "arc_challenge", "Which is a gas?", ["iron", "oxygen", "salt", "wood"], "B", "mcq", "science")
NUM = make_record("gsm8k_test_0001", "gsm8k", "2+3?", [], "5", "number", "math")
RECS = {MCQ["qid"]: MCQ, NUM["qid"]: NUM}
PP = {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}


# ---------------------------------------------------------------- parsing


def test_extract_json_variants():
    assert extract_json_object('{"answer": "B"}') == {"answer": "B"}
    assert extract_json_object('Sure!\n```json\n{"answer": "B"}\n```') == {"answer": "B"}
    assert extract_json_object('{"answer": "B", "confidence": 0.5,}') == {"answer": "B", "confidence": 0.5}
    assert extract_json_object("The {weird} answer is {\"answer\": \"C\"} ok") == {"answer": "C"}
    assert extract_json_object("no json here") is None


@pytest.mark.parametrize("raw,expected", [("B", "B"), ("b", "B"), ("(C)", "C"), ("D. wood", "D"), ("A) iron", "A"),
                                          ("E", None), ("Banana", None), (2, None), ("", None)])
def test_parse_letter(raw, expected):
    assert parse_letter(raw, 4) == expected


@pytest.mark.parametrize("raw,expected", [(0.7, 0.7), (70, 0.7), ("85%", 0.85), (1, 1.0), (-0.1, None), (150, None), ("hi", None)])
def test_parse_confidence(raw, expected):
    assert parse_confidence(raw) == pytest.approx(expected) if expected is not None else parse_confidence(raw) is None


def test_parse_distribution_mcq():
    d = parse_distribution({"a": 1, "B": 3}, MCQ)
    assert d == {"A": 0.25, "B": 0.75, "C": 0.0, "D": 0.0}
    assert parse_distribution([0.25, 0.25, 0.25, 0.25], MCQ)["C"] == 0.25
    assert parse_distribution({"Z": 1}, MCQ) is None
    assert parse_distribution({"A": 0, "B": 0}, MCQ) is None
    assert parse_distribution("uniform", MCQ) is None


def test_parse_distribution_number():
    assert parse_distribution({"5": 0.6, "5.0": 0.2, "6": 0.2}, NUM) == pytest.approx({"5": 0.8, "6": 0.2})


def test_parse_answer_mcq_and_number():
    p = parse_answer(answer_json("B", 0.9, PP), MCQ)
    assert (p.answer, p.confidence, p.error) == ("B", 0.9, None)
    assert p.peer_prediction == pytest.approx(PP)
    assert parse_answer('{"answer": "1,000"}', NUM).answer == "1000"
    assert parse_answer('{"answer": "F"}', MCQ).error == "bad_answer"
    assert parse_answer("I think B", MCQ).error == "no_json"


def test_system_prompt_mentions_letters_and_is_shared():
    sp = system_prompt(MCQ)
    assert '"A": <number>, "B": <number>, "C": <number>, "D": <number>' in sp and "OTHER council members" in sp
    assert "<a single number>" in system_prompt(NUM)


# ---------------------------------------------------------------- member behaviour


def _member(reply, probs=None, **kw):
    return CouncilMember("fake", FakeBackend(reply, probs), **kw)


def test_respond_valid_first_try():
    m = _member(lambda conv: answer_json("B", 0.8, PP))
    [[out]] = m.answer_prompts([apply_condition(MCQ, "neutral")], RECS)
    assert out.valid and out.answer == "B" and not out.retried
    assert out.peer_prediction_source == "model"
    assert m.backend.chat_calls[0][0][0]["role"] == "system"
    assert m.stats.invalid_rate == 0


def test_retry_once_then_valid():
    def reply(conv):
        return answer_json("C", 0.6, PP) if last_user(conv) == RETRY_MESSAGE else "I'd say C."

    m = _member(reply)
    [[out]] = m.answer_prompts([apply_condition(MCQ, "neutral")], RECS)
    assert out.valid and out.retried and out.answer == "C"
    assert m.stats.first_try_invalid_rate == 1.0 and m.stats.invalid_rate == 0.0


def test_invalid_after_retry_is_marked_and_counted():
    m = _member(lambda conv: "no idea")
    outs = m.answer_prompts([apply_condition(MCQ, "neutral"), apply_condition(NUM, "neutral")], RECS)
    assert all(not o[0].valid and o[0].answer is None and o[0].error == "no_json" for o in outs)
    assert m.stats.invalid_rate == 1.0 and m.stats.summary()["over_limit"]
    assert len(m.backend.chat_calls) == 2  # one batch + one retry batch


def test_token_prob_fallback_for_missing_peer_prediction():
    m = _member(lambda conv: answer_json("B", 0.8), probs=lambda conv: {"A": 0.2, "B": 0.6})
    [[out]] = m.answer_prompts([apply_condition(MCQ, "neutral")], RECS)
    assert out.peer_prediction_source == "token_probs"
    assert out.peer_prediction == pytest.approx({"A": 0.25, "B": 0.75, "C": 0.0, "D": 0.0})
    assert m.stats.pp_fallback == 1


def test_no_fallback_for_numbers_or_when_disabled():
    m = _member(lambda conv: answer_json("5", 0.8), probs=lambda conv: {"A": 1.0})
    [[out]] = m.answer_prompts([apply_condition(NUM, "neutral")], RECS)
    assert out.peer_prediction is None and out.peer_prediction_source == "none"
    assert m.backend.prob_calls == 0
    m2 = _member(lambda conv: answer_json("B"), probs=lambda conv: {"A": 1.0}, token_prob_fallback=False)
    [[out2]] = m2.answer_prompts([apply_condition(MCQ, "neutral")], RECS)
    assert out2.peer_prediction is None and m2.backend.prob_calls == 0


def test_always_option_probs_keeps_model_prediction():
    m = _member(lambda conv: answer_json("B", 0.8, PP), probs=lambda conv: {"B": 1.0}, always_option_probs=True)
    [[out]] = m.answer_prompts([apply_condition(MCQ, "neutral")], RECS)
    assert out.peer_prediction_source == "model" and out.option_probs["B"] == 1.0


def test_are_you_sure_runs_second_turn():
    def reply(conv):
        return answer_json("A") if last_user(conv) == ARE_YOU_SURE else answer_json("B")

    m = _member(reply)
    [outs] = m.answer_prompts([apply_condition(MCQ, "are_you_sure")], RECS)
    assert [o.stage for o in outs] == ["stage1", "stage1_turn2"]
    assert [o.answer for o in outs] == ["B", "A"]
    turn2_conv = m.backend.chat_calls[1][0]
    assert [x["role"] for x in turn2_conv] == ["system", "user", "assistant", "user"]


def test_outputs_never_contain_gold():
    m = _member(lambda conv: answer_json("B", 0.8, PP))
    [[out]] = m.answer_prompts([apply_condition(MCQ, "neutral")], RECS)
    assert "gold" not in out.to_dict()


def test_load_members_and_write_outputs(tmp_path):
    cfg = {"generation": {"temperature": 0.7}, "members": [{"name": "m1", "backend": {"model": "x"}}]}
    members = load_members(cfg, backend_factory=lambda **kw: FakeBackend(lambda c: answer_json("B")))
    assert members[0].name == "m1" and members[0].temperature == 0.7
    [[out]] = members[0].answer_prompts([apply_condition(MCQ, "neutral")], RECS)
    write_outputs([out], tmp_path / "o.jsonl")
    assert json.loads((tmp_path / "o.jsonl").read_text(encoding="utf-8"))["answer"] == "B"


# ---------------------------------------------------------------- real near-miss replies (local run, 2026-11-20)

LLAMA_MISSING_BRACE = (
    '{"reason": "Safety.", "answer": "C", "confidence": 0.9, "peer_prediction": '
    '{"A": "x", "B": "y", "C": "z", "D": "w"}'
)
GEMMA_MISSING_COLON = (
    '```json\n{\n  "reason": "Photosynthesis.",\n  "answer": "C",\n  "confidence": 0.95,\n'
    '  "peer_prediction"\n  {"A": "False", "B": "False", "C": "True", "D": "False"}\n}\n```'
)


def test_repairs_real_near_miss_replies():
    for raw in (LLAMA_MISSING_BRACE, GEMMA_MISSING_COLON):
        p = parse_answer(raw, MCQ)
        assert p.error is None and p.answer == "C"
        assert p.peer_prediction is None  # option text / booleans are not probabilities


def test_prefers_object_with_answer_over_nested_dict():
    # unrepairable outer object (missing comma): falls back to the inner dict, which has no answer
    raw = '{"answer": "B" "confidence": 0.5, "peer_prediction": {"A": 1}}'
    assert parse_answer(raw, MCQ).error == "bad_answer"
    assert extract_json_object('{"x": 1} then {"answer": "D"}') == {"answer": "D"}


class FakeOOM(Exception):
    pass


def _oom_above(limit, calls):
    """fn that fails with FakeOOM when a chunk is bigger than `limit`; doubles each item."""

    def fn(start, chunk):
        calls.append((start, len(chunk)))
        if len(chunk) > limit:
            raise FakeOOM()
        return [x * 2 for x in chunk]

    return fn


def test_run_in_batches_plain():
    calls = []
    out = run_in_batches(list(range(10)), 4, _oom_above(99, calls), lambda e: isinstance(e, FakeOOM))
    assert out == [x * 2 for x in range(10)]
    assert calls == [(0, 4), (4, 4), (8, 2)]


def test_run_in_batches_halves_on_oom_and_keeps_order():
    calls, sizes = [], []
    out = run_in_batches(
        list(range(10)), 8, _oom_above(3, calls), lambda e: isinstance(e, FakeOOM), on_oom=sizes.append
    )
    assert out == [x * 2 for x in range(10)], "every item exactly once, in order"
    assert sizes == [4, 2], "8 -> 4 -> 2, then the smaller size is kept"
    assert calls[:3] == [(0, 8), (0, 4), (0, 2)] and all(n <= 2 for _, n in calls[3:])


def test_run_in_batches_reraises():
    with pytest.raises(FakeOOM):  # still OOM at batch size 1
        run_in_batches([1, 2], 2, _oom_above(0, []), lambda e: isinstance(e, FakeOOM))
    with pytest.raises(ValueError):  # other errors are never retried

        def bad(start, chunk):
            raise ValueError("boom")

        run_in_batches([1, 2], 2, bad, lambda e: isinstance(e, FakeOOM))
