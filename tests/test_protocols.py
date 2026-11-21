import json
import re

import pytest

from council.members import CouncilMember
from council.protocols import (
    DEBATE_MESSAGE,
    RANK_SYSTEM,
    debate,
    independent,
    karpathy_style,
    parse_ranking,
    run_protocol,
)
from data.build_questions import make_record
from data.pressure import ARE_YOU_SURE, apply_condition
from tests.fakes import FakeBackend, answer_json, last_user

Q1 = make_record("arc_1", "arc_challenge", "Which is a gas?", ["iron", "oxygen", "salt", "wood"], "B", "mcq", "science")
Q2 = make_record("arc_2", "arc_challenge", "Which is a metal?", ["iron", "oxygen", "salt", "wood"], "A", "mcq", "science")
RECS = {r["qid"]: r for r in (Q1, Q2)}
PROMPTS = [apply_condition(Q1, "neutral"), apply_condition(Q2, "user_wrong")]


def stubborn(name: str, answer: str) -> CouncilMember:
    """Always gives `answer`; ranks responses by putting any response that says B first."""

    def reply(conv):
        if conv[0]["content"] == RANK_SYSTEM:
            labels = re.findall(r"Response (\d+):\nAnswer: (\S+)", last_user(conv))
            ranked = sorted(labels, key=lambda x: x[1] != "B")
            return json.dumps({"reason": "r", "ranking": [int(n) for n, _ in ranked]})
        return answer_json(answer)

    return CouncilMember(name, FakeBackend(reply))


def follower(name: str, first: str) -> CouncilMember:
    """Answers `first`, then switches to the majority answer it is shown in debate."""

    def reply(conv):
        msg = last_user(conv)
        if msg.startswith("These are the current answers"):
            shown = re.findall(r"Answer: (\S+)", msg)
            return answer_json(max(set(shown), key=shown.count))
        return answer_json(first)

    return CouncilMember(name, FakeBackend(reply))


def test_independent_shapes():
    members = [stubborn("a", "B"), stubborn("b", "C"), stubborn("c", "B")]
    res = independent(PROMPTS, RECS, members)
    assert [r.qid for r in res] == ["arc_1", "arc_2"]
    assert [o["answer"] for o in res[0].stages["stage1"]] == ["B", "C", "B"]
    assert [o["member"] for o in res[0].final_stage()] == ["a", "b", "c"]
    assert res[1].condition == "user_wrong"
    # one batched call per member for all questions
    assert all(len(m.backend.chat_calls) == 1 and len(m.backend.chat_calls[0]) == 2 for m in members)


def test_independent_are_you_sure_keeps_initial():
    def caver(conv):
        return answer_json("D") if last_user(conv) == ARE_YOU_SURE else answer_json("B")

    members = [CouncilMember("x", FakeBackend(caver)), stubborn("y", "B")]
    [res] = independent([apply_condition(Q1, "are_you_sure")], RECS, members)
    assert [o["answer"] for o in res.stages["stage1_initial"]] == ["B", "B"]
    assert [o["answer"] for o in res.stages["stage1"]] == ["D", "B"]


def test_debate_members_can_revise():
    members = [stubborn("a", "B"), stubborn("b", "B"), follower("c", "D")]
    res = debate(PROMPTS, RECS, members, rounds=1)
    assert res[0].protocol == "debate"
    assert res[0].stages["stage1"][2]["answer"] == "D"
    assert res[0].stages["debate_round1"][2]["answer"] == "B"
    assert res[0].final_stage() == res[0].stages["debate_round1"]
    # c saw only the OTHER two members
    debate_msg = last_user(members[2].backend.chat_calls[1][0])
    assert debate_msg.count("Answer:") == 2 and "D" not in re.findall(r"Answer: (\S+)", debate_msg)


def test_debate_two_rounds_and_bad_rounds():
    members = [stubborn("a", "B"), follower("c", "D")]
    res = debate(PROMPTS[:1], RECS, members, rounds=2)
    assert {"stage1", "debate_round1", "debate_round2"} <= set(res[0].stages)
    with pytest.raises(ValueError):
        debate(PROMPTS, RECS, members, rounds=3)


@pytest.mark.parametrize("text,expected", [
    ('{"ranking": [2, 1, 3]}', [2, 1, 3]),
    ('{"ranking": ["Response 3", "Response 1", "Response 2"]}', [3, 1, 2]),
    ('{"ranking": [1, 1, 2]}', None),
    ('{"ranking": [1, 2]}', None),
    ("1 > 2 > 3", None),
])
def test_parse_ranking(text, expected):
    assert parse_ranking(text, 3)[0] == expected


def test_karpathy_style_ranks_and_chairman():
    members = [stubborn("a", "B"), stubborn("b", "C"), stubborn("c", "B")]
    res = karpathy_style(PROMPTS, RECS, members)
    r = res[0]
    assert r.protocol == "karpathy_style" and r.meta["chairman"] == "a"
    assert len(r.stages["stage2_rank"]) == 3 and all(e["valid"] for e in r.stages["stage2_rank"])
    # every ranker put member b (the only C answer) last, whatever order it saw
    assert all(e["ranking_members"][-1] == "b" for e in r.stages["stage2_rank"])
    assert r.meta["avg_rank"]["b"] == 3.0
    [chair] = r.stages["stage3_chairman"]
    assert chair["member"] == "a" and chair["answer"] == "B"
    # aggregators use member answers, not the chairman
    assert len(r.final_stage()) == 3


def test_karpathy_anonymises_member_names():
    members = [stubborn("qwen", "B"), stubborn("llama", "C")]
    karpathy_style(PROMPTS, RECS, members)
    for m in members:
        for conv in m.backend.chat_calls[1]:  # ranking call
            assert "qwen" not in str(conv) and "llama" not in str(conv)


def test_run_protocol_dispatch():
    members = [stubborn("a", "B"), stubborn("b", "B")]
    assert run_protocol("independent", PROMPTS, RECS, members)[0].protocol == "independent"
    assert run_protocol("debate", PROMPTS, RECS, members, rounds=1)[0].protocol == "debate"
    with pytest.raises(ValueError):
        run_protocol("vote", PROMPTS, RECS, members)


def test_debate_message_template():
    assert "{others}" in DEBATE_MESSAGE


def test_stage1_cache_shares_first_answers_across_protocols():
    members = [stubborn("a", "B"), stubborn("b", "C"), stubborn("c", "B")]
    cache: dict = {}
    ind = run_protocol("independent", PROMPTS, RECS, members, stage1_cache=cache)
    deb = run_protocol("debate", PROMPTS, RECS, members, stage1_cache=cache)
    kar = run_protocol("karpathy_style", PROMPTS, RECS, members, stage1_cache=cache)
    assert ind[0].stages["stage1"] == deb[0].stages["stage1"] == kar[0].stages["stage1"]
    assert ind[0].protocol == "independent" and "debate_round1" not in ind[0].stages  # no cross-talk
    # stage 1 generated once per member; debate and ranking each add one call
    stage1_calls = [c for c in members[0].backend.chat_calls if c[0][0]["content"] != RANK_SYSTEM]
    assert len(stage1_calls) == 3  # stage1 once + debate round + chairman (members[0] chairs)
