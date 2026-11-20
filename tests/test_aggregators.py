import pytest

from council.aggregators import (
    chairman,
    confidence_weighted,
    dawid_skene,
    inverse_surprising_popularity,
    majority_vote,
    optimal_weight,
    pick_best,
    surprisingly_popular,
)


def out(member, answer, conf=None, pp=None, valid=True):
    return {"member": member, "answer": answer if valid else None, "confidence": conf,
            "peer_prediction": pp, "valid": valid}


def test_majority_vote_and_invalid_ignored():
    outs = [out("a", "B"), out("b", "B"), out("c", "C"), out("d", "C", valid=False)]
    assert majority_vote(outs) == "B"
    assert majority_vote([out("a", "B", valid=False)]) is None


def test_tie_break_is_seeded_not_alphabetical():
    outs = [out("a", "A"), out("b", "B")]
    picks = {majority_vote(outs, tie_key=f"q{i}") for i in range(50)}
    assert picks == {"A", "B"}
    assert majority_vote(outs, "q1") == majority_vote(outs, "q1")


def test_confidence_weighted():
    outs = [out("a", "A", 0.9), out("b", "B", 0.3), out("c", "B", 0.4)]
    assert confidence_weighted(outs) == "A"  # 0.9 > 0.7
    assert majority_vote(outs) == "B"
    assert confidence_weighted([out("a", "A"), out("b", "B", 0.6)]) == "B"  # missing -> 0.5


def test_surprisingly_popular_rescues_informed_minority():
    # Classic SP example: everyone predicts most will say A, but B is more popular than predicted.
    pp = {"A": 0.8, "B": 0.2}
    outs = [out("a", "A", pp=pp), out("b", "A", pp=pp), out("c", "B", pp=pp), out("d", "B", pp=pp), out("e", "A", pp=pp)]
    # actual A=0.6 (pred 0.8 -> -0.2), B=0.4 (pred 0.2 -> +0.2)
    assert majority_vote(outs) == "A"
    assert surprisingly_popular(outs) == "B"


def test_surprisingly_popular_falls_back_to_majority():
    outs = [out("a", "A"), out("b", "A"), out("c", "B")]
    assert surprisingly_popular(outs) == "A"
    assert surprisingly_popular([]) is None


def test_dawid_skene_downweights_unreliable_member():
    # r1 always right, r2 right 4/5 of the time, s always wrong.
    questions = []
    for i in range(30):
        truth = "ABCD"[i % 4]
        wrong = "ABCD"[(i + 1) % 4]
        questions.append([out("r1", truth), out("r2", truth if i % 5 else wrong), out("s", wrong)])
    answers, acc = dawid_skene(questions, n_classes=[4] * 30)
    assert acc["r1"] > acc["r2"] > acc["s"]
    assert all(a == "ABCD"[i % 4] for i, a in enumerate(answers))


def test_dawid_skene_breaks_two_way_ties_using_reliability():
    # r is always with the majority elsewhere; on the 1-1 tie question DS should trust r.
    qs = [[out("r", "A"), out("x", "A"), out("y", "A")] for _ in range(10)]
    qs += [[out("r", "B"), out("x", "B"), out("y", "C")] for _ in range(10)]  # y unreliable
    qs.append([out("r", "D"), out("y", "C")])
    answers, acc = dawid_skene(qs, n_classes=[4] * len(qs))
    assert acc["r"] > acc["y"]
    assert answers[-1] == "D"


def test_dawid_skene_handles_empty_question():
    answers, _ = dawid_skene([[out("a", "A")], [out("a", None, valid=False)]])
    assert answers[1] is None


def test_chairman():
    res = {"stages": {"stage3_chairman": [out("a", "C")]}}
    assert chairman(res) == "C"
    assert chairman({"stages": {}}) is None


def test_isp_ow_are_stubs():
    with pytest.raises(NotImplementedError):
        inverse_surprising_popularity([])
    with pytest.raises(NotImplementedError):
        optimal_weight([])


def test_pick_best_empty():
    assert pick_best({}) is None
