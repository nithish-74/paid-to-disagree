import math
import random

import pytest

from rewards.bts import actual_frequency, bts_info, kl, predicted_frequency, prediction_score, smooth
from rewards.ca import agree, ca_batch, ca_score
from rewards.dmi import dmi_batch
from rewards.pts import pts_score

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


def test_unimplemented_mechanisms_are_stubs():
    with pytest.raises(NotImplementedError):
        pts_score()
    with pytest.raises(NotImplementedError):
        dmi_batch()
