"""Every ablation / baseline training config must load and be internally consistent, so a run
started from kaggle/c7_train.ipynb on any account cannot fail on a typo hours in."""

from pathlib import Path

import pytest
import yaml

from rewards.combined import RewardConfig
from train.grpo_roundrobin import grpo_args, member_cfg, train_settings, training_population

TRAIN_CONFIGS = sorted(Path("configs/ablations").glob("train_*.yaml")) + [Path("configs/train.yaml"), Path("configs/train_b9.yaml")]


def load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.stem)
def test_train_config_is_consistent(path, tmp_path):
    cfg = load(path)
    RewardConfig.from_dict(load(cfg["reward_config"]))  # unknown keys or a bad mechanism raise
    council = load(cfg["council_config"])
    for name in cfg["rotation"]:
        train_settings(member_cfg(council, name))  # every rotation member exists and has a known dtype
    training_population(cfg)
    grpo_args(cfg, tmp_path)  # batch x accumulation is a multiple of the group size
    assert set(cfg["conditions"]) <= {"neutral", "user_wrong", "user_right", "are_you_sure"}
    for f in cfg["train_data"]:
        assert f.startswith("data/processed/") and f.endswith(".jsonl")


def test_experiment_names_unique():
    names = [load(p)["experiment"] for p in TRAIN_CONFIGS]
    assert len(names) == len(set(names)), names


def test_ablations_change_what_they_say():
    main = load("configs/train.yaml")
    main_reward = load("configs/reward.yaml")
    assert load("configs/ablations/reward_no_flip.yaml")["l5_flip"] == 0.0
    assert load("configs/ablations/reward_mech_dmi.yaml")["pp_mechanism"] == "dmi"
    assert load("configs/ablations/train_abl5_g16.yaml")["grpo"]["num_generations"] == 16
    assert "user_right" not in load("configs/ablations/train_abl6_no_user_right.yaml")["conditions"]
    assert load("configs/ablations/train_abl3_single_model.yaml")["population"] == "single_model"
    assert len(load("configs/ablations/train_abl7_size4.yaml")["rotation"]) == 4
    assert len(load("configs/ablations/train_abl7_size2.yaml")["rotation"]) == 2
    homo = load("configs/ablations/council_homogeneous.yaml")
    assert {m["backend"]["model"] for m in homo["members"]} == {"Qwen/Qwen2.5-1.5B-Instruct"}
    # every ablation keeps the main run's data and steps unless it is the thing being varied
    for p in Path("configs/ablations").glob("train_*.yaml"):
        cfg = load(p)
        assert cfg["train_data"] == main["train_data"] and cfg["grpo"]["max_steps"] == main["grpo"]["max_steps"]
    # only B10 may switch the oracle (labels) on
    for p in Path("configs/ablations").glob("reward_*.yaml"):
        assert load(p)["oracle"] is (p.name == "reward_b10_correctness.yaml")
    assert main_reward["oracle"] is False
