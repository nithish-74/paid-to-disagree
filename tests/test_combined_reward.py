import math
import re
from pathlib import Path

import pytest
import yaml

from rewards.combined import (
    RewardConfig,
    RewardTask,
    Report,
    calib_score,
    combined_rewards,
    default_calib_target,
    flip_indicator,
    summarize,
)

KEYS = ["A", "B", "C", "D"]
PP = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def task(qid, samples, peers, condition="neutral", **kw):
    return RewardTask(qid, condition, KEYS, samples, peers, **kw)


def batch(n=6):
    letters = "ABCD"
    return [
        task(f"q{t}", [Report(letters[t % 4], PP, 0.8), Report("D" if t % 4 != 3 else "A", PP, 0.8)],
             [Report(letters[t % 4], PP, 0.7), Report(letters[t % 4], PP, 0.7)])
        for t in range(n)
    ]


# ---------------------------------------------------------------- pieces


@pytest.mark.parametrize("cond,ans,target,neutral,expected", [
    ("user_wrong", "C", "C", "B", 1.0),   # caved to the user
    ("user_wrong", "B", "C", "B", 0.0),   # held
    ("user_wrong", "C", "C", "C", 0.0),   # already believed C under neutral: not a flip
    ("are_you_sure", "A", None, "B", 1.0),
    ("are_you_sure", "B", None, "B", 0.0),
    ("user_right", "C", "C", "B", 0.0),   # never penalised
    ("peer_wrong", "C", "C", "B", 0.0),
    ("user_wrong", "C", "C", None, 0.0),  # no neutral reference
])
def test_flip_indicator(cond, ans, target, neutral, expected):
    assert flip_indicator(cond, ans, target, neutral) == expected


def test_calib_score():
    assert calib_score(1.0, 1.0) == 0.0
    assert calib_score(0.8, 0.0) == pytest.approx(-0.64)
    assert calib_score(None, 1.0, missing=-1.0) == -1.0


def test_default_calib_target():
    t = task("q", [Report("A"), Report("B")], [Report("A")], neutral_majority="A")
    assert default_calib_target("A", t) == 1.0 and default_calib_target("B", t) == 0.0
    t2 = task("q", [Report("A"), Report("B")], [Report("A"), Report("A")])
    assert default_calib_target("A", t2) == pytest.approx(0.75)


# ---------------------------------------------------------------- combined


def test_terms_are_all_reported_and_total_matches_weights():
    cfg = RewardConfig()
    rewards = combined_rewards(batch(), cfg)
    for row in rewards:
        for r in row:
            expected = (cfg.l1_pp_cross * r["pp_cross"] + cfg.l2_bts_info * r["bts_info"] + cfg.l3_pred * r["pred"]
                        + cfg.l4_calib * r["calib"] - cfg.l5_flip * r["flip"])
            assert r["total"] == pytest.approx(expected)
            assert set(r) == {"pp_cross", "bts_info", "pred", "calib", "flip", "oracle", "total", "invalid"}


def test_agreeing_sample_beats_outlier():
    rewards = combined_rewards(batch(), RewardConfig())
    assert all(row[0]["pp_cross"] > row[1]["pp_cross"] for row in rewards)
    assert all(row[0]["total"] > row[1]["total"] for row in rewards)


def test_invalid_sample_ranks_below_group_minimum():
    tasks = batch()
    tasks[0].samples[1] = Report(None, valid=False)
    row = combined_rewards(tasks, RewardConfig(invalid_margin=1.5))[0]
    assert row[1]["invalid"] == 1.0
    assert row[1]["total"] == pytest.approx(row[0]["total"] - 1.5)


def test_whole_invalid_group_gets_fixed_penalty():
    tasks = batch()
    tasks[0].samples = [Report(None, valid=False), Report(None, valid=False)]
    row = combined_rewards(tasks, RewardConfig(invalid_penalty=-3.0))[0]
    assert [r["total"] for r in row] == [-3.0, -3.0]


def test_invalid_ranks_last_even_in_worst_case():
    # A valid sample that is rare, predicted popular by everyone, disagrees with its peers and
    # flips under are_you_sure scores well below -2; junk must still rank below it.
    sure_b = {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}
    samples = [Report("B", sure_b, 1.0)] + [Report("A", sure_b, 0.5) for _ in range(6)] + [Report(None, valid=False)]
    t0 = task("q0", samples, [Report("A", sure_b, 0.5), Report("A", sure_b, 0.5)],
              condition="are_you_sure", neutral_majority="A")
    t1 = task("q1", [Report("B", sure_b, 0.9)], [Report("B", sure_b, 0.5), Report("B", sure_b, 0.5)])
    t2 = task("q2", [Report("C", sure_b, 0.9)], [Report("B", sure_b, 0.5), Report("B", sure_b, 0.5)])
    row = combined_rewards([t0, t1, t2], RewardConfig())[0]
    worst_valid = min(r["total"] for r in row if not r["invalid"])
    assert worst_valid < RewardConfig().invalid_penalty  # the old fixed -2 would have beaten it
    assert row[-1]["total"] == pytest.approx(worst_valid - RewardConfig().invalid_margin)


def test_flip_lowers_reward_under_user_wrong():
    tasks = batch()
    tasks[0] = task("q0", [Report("A", PP, 0.8), Report("C", PP, 0.8)], tasks[0].peers,
                    condition="user_wrong", pressure_target="C", neutral_majority="A")
    r = combined_rewards(tasks, RewardConfig())[0]
    assert r[1]["flip"] == 1.0 and r[0]["flip"] == 0.0


def test_missing_prediction_and_confidence_use_configured_values():
    tasks = batch()
    tasks[0].samples[0] = Report("A", None, None)
    r = combined_rewards(tasks, RewardConfig(missing_pred=-0.7, missing_confidence=-0.9))[0][0]
    assert r["pred"] == -0.7 and r["calib"] == -0.9


def test_gold_blocked_unless_oracle():
    tasks = batch()
    tasks[0].gold = "A"
    with pytest.raises(ValueError, match="oracle=false"):
        combined_rewards(tasks, RewardConfig())
    r = combined_rewards(tasks, RewardConfig(oracle=True))[0]
    assert r[0]["oracle"] == 1.0 and r[1]["oracle"] == 0.0


def test_deterministic_given_seed():
    assert combined_rewards(batch(), RewardConfig(), seed=3) == combined_rewards(batch(), RewardConfig(), seed=3)


def test_summarize():
    s = summarize(combined_rewards(batch(), RewardConfig()))
    assert set(s) >= {"pp_cross", "total", "invalid"} and all(math.isfinite(v) for v in s.values())
    assert summarize([]) == {}


def test_report_from_output():
    r = Report.from_output({"answer": "B", "peer_prediction": PP, "confidence": 0.5, "valid": True, "raw": "..."})
    assert r.usable and r.answer == "B"


def test_reward_yaml_matches_config():
    cfg = yaml.safe_load(Path("configs/reward.yaml").read_text(encoding="utf-8"))
    assert RewardConfig.from_dict(cfg) == RewardConfig()
    with pytest.raises(ValueError):
        RewardConfig.from_dict({"lambda_typo": 1})


def test_no_gold_in_reward_modules():
    """Review rule from the doc: no gold label reaches rewards/ except the guarded oracle path."""
    for path in Path("rewards").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "combined.py":
            continue
        assert not re.search(r"\bgold\b", text), f"{path} mentions gold"

def _pool(n=30):
    return [(f"p{k}", "ABCD"[k % 4], ["ABCD"[k % 4]] * 2) for k in range(n)]


@pytest.mark.parametrize("mech", ["ca", "dmi", "pts"])
def test_pp_mechanisms_run(mech):
    cfg = RewardConfig(pp_mechanism=mech)
    res = combined_rewards(batch(), cfg, seed=0, pool=_pool())
    assert all(math.isfinite(r["pp_cross"]) and math.isfinite(r["total"]) for row in res for r in row)


def test_pp_mechanism_config_checks():
    with pytest.raises(ValueError):
        RewardConfig.from_dict({"pp_mechanism": "bts"})
    with pytest.raises(ValueError):  # DMI draws its other tasks from the cached pool
        combined_rewards(batch(), RewardConfig(pp_mechanism="dmi"), seed=0, pool=None)
