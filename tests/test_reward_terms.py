import math
import random

import pytest

from rewards.bts import actual_frequency, bts_info, kl, predicted_frequency, prediction_score, smooth
from rewards.ca import agree, ca_batch, ca_score
from rewards.dmi import det, dmi_batch, dmi_payment
from rewards.pts import pts_batch, pts_score, report_distribution

KEYS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------- CA


def test_agree_and_ca_score():
    assert agree("A", "A") == 1.0 and agree("A", "B") == 0.0 and agree(None, None) == 0.0
    assert ca_score("A", "A", "B", "C") == 1.0
    assert ca_score("A", "A", "B", "B") == 0.0
    assert ca_score("A", "B", "C", "C") == -1.0


def test_ca_constant_answer_earns_zero():
    """Collusion check: if everyone always says 'A', CA pays exactly 0."""
    T = 6
    scores = ca_batch([["A"] * 4 for _ in range(T)], [["A", "A"] for _ in range(T)], random.Random(0))
    assert all(s == 0.0 for row in scores for s in row)


def test_ca_informative_agreement_pays():
    """Answers that track the question (and agree with peers) earn ~ 1 - P(chance agreement)."""
    truth = ["A", "B", "C", "D"] * 5
    own = [[t] * 4 for t in truth]
    peers = [[t, t] for t in truth]
    scores = ca_batch(own, peers, random.Random(0))
    mean = sum(s for row in scores for s in row) / sum(len(r) for r in scores)
    assert 0.6 < mean <= 1.0


def test_ca_disagreeing_sample_scores_lower():
    truth = ["A", "B", "C", "D"] * 3
    own = [[t, "Z"] for t in truth]  # sample 0 agrees with peers, sample 1 never does
    peers = [[t, t] for t in truth]
    scores = ca_batch(own, peers, random.Random(1))
    assert all(row[0] > row[1] for row in scores)


def test_ca_validates_shapes():
    with pytest.raises(ValueError):
        ca_batch([["A"]], [], random.Random(0))


# ---------------------------------------------------------------- BTS


def test_smooth_and_frequencies():
    s = smooth({"A": 1.0}, KEYS, eps=0.0)
    assert s == {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0}
    assert sum(smooth({"A": 1.0}, KEYS, 0.1).values()) == pytest.approx(1.0)
    assert actual_frequency(["A", "A", "B", None]) == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    assert actual_frequency([None]) == {}


def test_predicted_frequency_means():
    p1, p2 = {"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}
    assert predicted_frequency([p1, p2], ["A", "B"], "arithmetic", eps=0) == pytest.approx({"A": 0.5, "B": 0.5})
    geo = predicted_frequency([{"A": 0.8, "B": 0.2}, {"A": 0.2, "B": 0.8}], ["A", "B"], "geometric", eps=0)
    assert geo == pytest.approx({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError):
        predicted_frequency([p1], ["A"], "median")


def test_bts_info_rewards_surprisingly_common_answer():
    # Everyone predicts 80% will say A, but B is chosen by 40%.
    preds = [{"A": 0.8, "B": 0.2}] * 5
    answers = ["A", "A", "A", "B", "B"]
    info_b = bts_info("B", answers, preds, ["A", "B"], mean="arithmetic", eps=1e-9)
    info_a = bts_info("A", answers, preds, ["A", "B"], mean="arithmetic", eps=1e-9)
    assert info_b == pytest.approx(math.log(0.4 / 0.2))
    assert info_a == pytest.approx(math.log(0.6 / 0.8))
    assert info_b > 0 > info_a


def test_bts_info_degenerate_cases():
    assert bts_info(None, ["A"], [{"A": 1}], KEYS) == 0.0
    assert bts_info("A", ["A"], [], KEYS) == 0.0


def test_kl_and_prediction_score():
    assert kl({"A": 1.0}, {"A": 1.0}) == 0.0
    perfect = prediction_score({"A": 0.5, "B": 0.5}, ["A", "B"], ["A", "B"], eps=1e-9)
    worse = prediction_score({"A": 0.9, "B": 0.1}, ["A", "B"], ["A", "B"], eps=1e-9)
    assert perfect == pytest.approx(0.0, abs=1e-6)
    assert worse == pytest.approx(-(0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)), rel=1e-6)
    assert prediction_score(None, ["A"], KEYS) is None
    assert prediction_score({"A": 1}, [None], KEYS) is None


def test_prediction_score_handles_unpredicted_numbers():
    # Number question: peers said "7", member only predicted "5" -> large but finite penalty.
    s = prediction_score({"5": 1.0}, ["7", "7"], [], eps=1e-3)
    assert s is not None and math.isfinite(s) and s < -5


# ---------------------------------------------------------------- DMI (Kong 2020, Section 5)


def test_det_exact():
    assert det([[1, 0], [0, 1]]) == 1
    assert det([[0, 1], [1, 0]]) == -1
    assert det([[2, 0, 1], [1, 3, 2], [1, 1, 1]]) == 0  # 2(3-2) - 0 + 1(1-3)
    assert det([[1, 2], [2, 4]]) == 0
    assert det([[2 if i == j else 0 for j in range(4)] for i in range(4)]) == 16


def test_dmi_payment_truthful_vs_uninformative():
    keys = ["A", "B"]
    answers = list("ABABABAB")
    agree_p = dmi_payment(answers, [[a] for a in answers], keys, split=4, normalise=False)
    assert agree_p == 4 * 4  # each half: diag(2, 2), det 4
    constant = dmi_payment(["A"] * 8, [[a] for a in answers], keys, split=4, normalise=False)
    assert constant == 0  # answering the same thing everywhere earns nothing
    # Kong's Figure 1 point: negatively correlated answers make both determinants negative,
    # but their product is still positive.
    flipped = ["B" if a == "A" else "A" for a in answers]
    assert dmi_payment(flipped, [[a] for a in answers], keys, split=4, normalise=False) == 16


def test_dmi_normalisation_and_errors():
    keys = ["A", "B"]
    answers = list("ABABABAB")
    q = dmi_payment(answers, [[a, a] for a in answers], keys, split=4)
    # p = 2 peers * 16 = 32; norm = (n - 1) = 2 * (2!)^2 * C(4, 2) * C(4, 2) = 2 * 4 * 6 * 6 = 288
    assert q == pytest.approx(32 / 288)
    with pytest.raises(ValueError):
        dmi_payment(answers[:3], [[a] for a in answers[:3]], keys, split=1)  # |T1| < C


def _dmi_pool(n=30):
    letters = "AB"
    return [(f"p{k}", letters[k % 2], [letters[k % 2], letters[k % 2]]) for k in range(n)]


def test_dmi_batch_rewards_agreeing_with_informative_peers():
    pool = _dmi_pool()
    own = [["A", "B"]]  # sample 0 agrees with both peers, sample 1 does not
    peers = [["A", "A"]]
    agree_scores, disagree_scores = [], []
    for seed in range(40):
        s = dmi_batch(own, peers, [["A", "B"]], random.Random(seed), pool, ["q0"], n_tasks=8)
        agree_scores.append(s[0][0])
        disagree_scores.append(s[0][1])
    assert sum(agree_scores) > sum(disagree_scores)
    numeric = dmi_batch([["7"]], [["7", "7"]], [[]], random.Random(0), pool, ["q1"], n_tasks=8)
    assert numeric == [[0.0]]


def test_dmi_batch_needs_enough_tasks():
    with pytest.raises(ValueError):
        dmi_batch([["A"]], [["A"]], [["A", "B"]], random.Random(0), _dmi_pool(3), ["q0"], n_tasks=8)
    with pytest.raises(ValueError):
        dmi_batch([["A"]], [["A"]], [list("ABCD")], random.Random(0), _dmi_pool(), ["q0"], n_tasks=6)


# ---------------------------------------------------------------- PTS (Faltings et al., Def. 17)


def test_pts_score_definition():
    R = {k: 0.25 for k in "ABCD"}
    assert pts_score("B", "B", R) == pytest.approx(-1 + 4)  # f + C / R[r]
    assert pts_score("B", "C", R) == -1.0  # f(rr) only
    assert pts_score("B", "B", R, C=2.0, f=0.0) == pytest.approx(8.0)
    # f = -C: an agent who knows only R expects 0, whatever it reports (arbitrage-free setting)
    R2 = {"A": 0.5, "B": 0.3, "C": 0.2}
    for r in R2:
        assert sum(R2[rr] * pts_score(r, rr, R2) for rr in R2) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pts_score("A", "A", R, C=0.0)


def test_pts_rare_agreement_pays_more():
    R = report_distribution(list("AAAAAB"), list("AB"))  # counts + 1: A 6/8, B 2/8
    assert R == {"A": pytest.approx(0.75), "B": pytest.approx(0.25)}
    assert pts_score("B", "B", R) > pts_score("A", "A", R)


def test_pts_batch():
    own = [["A", "B", None], ["5"]]
    peers = [["A", None], ["5", "5"]]
    s = pts_batch(own, peers, [list("AB"), []], pool=[("p", "B", ["B", "B"])])
    R = report_distribution(["A", None, "5", "5", "B", "B"], list("AB"))
    assert s[0][0] == pytest.approx(pts_score("A", "A", R))  # only the valid peer counts
    assert s[0][1] == pytest.approx(-1.0) and s[0][2] == pytest.approx(-1.0)
    assert s[1] == [0.0]  # numeric task
