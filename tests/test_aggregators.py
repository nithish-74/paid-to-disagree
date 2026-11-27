import math

import pytest

from council.aggregators import (
    chairman,
    confidence_weighted,
    dawid_skene,
    inverse_surprising_popularity,
    isp_advantage,
    labelled_accuracies,
    majority_vote,
    model_second_order,
    optimal_weight,
    ow_i_accuracies,
    ow_l_accuracies,
    ow_weight,
    pick_best,
    second_order,
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


def _true_second_order(acc, labels):
    """P(A_i | A_j) from the paper's closed form (Appendix F.2) for given accuracies."""
    K = len(labels)
    return {(i, j): {(k, l): model_second_order(acc[i], acc[j], k == l, K) for k in labels for l in labels}
            for i in acc for j in acc if i != j}


def test_isp_paper_example_1():
    # Ai et al. Example 1 / Table 1: x1 = x2 = 1, x3 = x4 = 0.5, K = 2, true label s1.
    labels = ["A", "B"]
    P = _true_second_order({"m1": 1.0, "m2": 1.0, "m3": 0.5, "m4": 0.5}, labels)
    outs = [out("m1", "A"), out("m2", "A"), out("m3", "B"), out("m4", "B")]
    adv = isp_advantage(outs, P, labels)
    assert adv["A"] == pytest.approx(1 / 3)  # App. E.2: ISP's advantage of s1 is 1/3
    assert inverse_surprising_popularity(outs, P, labels) == "A"  # MV would tie, SP picks s2
    for case in (["A", "A", "A", "A"], ["A", "A", "A", "B"], ["A", "A", "B", "A"]):
        outs = [out(f"m{n + 1}", a) for n, a in enumerate(case)]
        assert inverse_surprising_popularity(outs, P, labels) == "A"


def test_isp_advantages_sum_to_zero_and_fallback():
    labels = list("ABCD")
    P = _true_second_order({"a": 0.8, "b": 0.6, "c": 0.5}, labels)
    outs = [out("a", "A"), out("b", "B"), out("c", "B")]
    assert sum(isp_advantage(outs, P, labels).values()) == pytest.approx(0.0)
    one = [out("a", "C"), out("b", "B", valid=False)]
    assert inverse_surprising_popularity(one, P, labels) == "C"  # < 2 valid -> majority vote


def test_second_order_counts():
    qs = [[out("a", "A"), out("b", "A")], [out("a", "A"), out("b", "B")], [out("a", "B"), out("b", "B")],
          [out("a", "C"), out("b", "C", valid=False)]]
    P = second_order(qs, list("ABC"))
    assert P[("a", "b")][("A", "A")] == 1.0
    assert P[("a", "b")][("A", "B")] == pytest.approx(0.5)
    assert ("A", "C") not in P[("a", "b")]  # b never validly said C: left out (1/K is used later)


def test_ow_weights_and_strong_member_wins():
    assert ow_weight(0.25, 4) == pytest.approx(0.0)  # chance level -> weight 0
    assert ow_weight(0.1, 4) == pytest.approx(0.0)  # below chance is clipped, i.e. excluded
    assert ow_weight(0.6, 2) == pytest.approx(math.log(0.6 / 0.4))  # K = 2: inverse logistic (Cor. 3.2)
    assert math.isfinite(ow_weight(1.0, 4))
    outs = [out("s", "A"), out("w1", "B"), out("w2", "B")]
    acc = {"s": 0.95, "w1": 0.6, "w2": 0.6}  # ln 57 = 4.04 > 2 * ln 4.5 = 3.01
    assert optimal_weight(outs, acc, K=4) == "A"
    assert majority_vote(outs) == "B"
    assert optimal_weight(outs, {"s": 0.95}, K=4) == "B"  # missing accuracy -> majority vote


def test_ow_l_recovers_accuracies():
    labels = list("ABCD")
    truth = {"a": 0.9, "b": 0.7, "c": 0.55}
    est = ow_l_accuracies(_true_second_order(truth, labels), labels)
    for m, x in truth.items():
        assert est[m] == pytest.approx(x, abs=0.02)


def test_ow_i_and_labelled_accuracies():
    qs = [[out("a", "A"), out("b", "B")], [out("a", "C"), out("b", "C")],
          [out("a", "D", valid=False), out("b", "D")]]
    assert ow_i_accuracies(qs, ["A", "C", None]) == {"a": 1.0, "b": 0.5}
    assert labelled_accuracies(qs, ["A", "C", "D"]) == {"a": pytest.approx(2 / 3), "b": pytest.approx(2 / 3)}


def test_pick_best_empty():
    assert pick_best({}) is None
