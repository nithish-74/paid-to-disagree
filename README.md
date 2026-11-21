# Paid to Disagree

Training a council of three small LLMs from different model families (Qwen2.5-1.5B, Llama-3.2-1B, Gemma-3-1B) with a label-free, cross-model peer-prediction reward (GRPO), to test whether it reduces sycophancy and makes members' errors less correlated.

## Layout

```
configs/    YAML config per experiment
data/       C1 question data (build_questions.py), C2 pressure conditions (pressure.py)
council/    C3 members, C4 protocols, C5 aggregators
rewards/    C6 peer-prediction reward terms
train/      C7 round-robin GRPO
eval/       C8 evaluation and metrics
analysis/   scripts that produce paper tables and plots
runs/       run outputs (not committed)
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; use .venv/bin/python on Linux/Kaggle
python -m data.build_questions --config configs/data.yaml   # downloads datasets -> data/processed/
python -m pytest
```

GPU parts (vLLM, TRL, Unsloth) run on Kaggle/Colab (T4, fp16).

## Phase 1 member check (Kaggle, GPU)

1. Accept the model licences on Hugging Face for `meta-llama/Llama-3.2-1B-Instruct` and `google/gemma-3-1b-it`, and add `HF_TOKEN` as a Kaggle secret.
2. In a GPU notebook:

```bash
git clone https://github.com/nithish-74/paid-to-disagree.git && cd paid-to-disagree
pip install -e . vllm
python -m data.build_questions --config configs/data.yaml
python -m council.check_members --data data/processed/arc_challenge.jsonl --n 50
```

Each member must show `invalid` under 5% on the dev set before moving on. Outputs go to `runs/<run_id>/`.

## Round-robin GRPO training (Kaggle, GPU)

One member is trained at a time against frozen peers; progress is kept in `runs/<experiment>/state.json`, so each step can run in a separate Kaggle session. Copy the `runs/<experiment>/` folder between sessions (e.g. as a Kaggle dataset).

```bash
pip install -e . unsloth vllm trl
python -m train.grpo_roundrobin cache  --config configs/train.yaml               # all members, once
python -m train.grpo_roundrobin train  --config configs/train.yaml               # next leg (e.g. round 1, qwen)
python -m train.grpo_roundrobin cache  --config configs/train.yaml --members qwen # refresh the member just trained
python -m train.grpo_roundrobin status --config configs/train.yaml               # what's done, what's next
```

Each leg writes `monitor.jsonl` (dev accuracy, answer entropy, flip rate every 200 steps) and `reward_terms.jsonl` (every reward term) to `runs/<experiment>/round<r>_<member>/`.

## Baseline on Kaggle (notebook)

`kaggle/baseline_eval.ipynb` does the whole untrained baseline in one go: install, model access
(HF_TOKEN secret or Unsloth's ungated mirrors), data, the Phase 1 member check, a 5-question smoke
eval, then seeds 0-2 on the full test splits (one seed per GPU on T4 x2), metrics and the
seed-combined tables, zipped as `eval_untrained.zip`. Open it on Kaggle via File -> Import Notebook,
set GPU T4 x2 and Internet on, run steps 1-5 interactively, then Save & Run All (Commit).

## Evaluation

```bash
python -m eval.run_eval --config configs/eval.yaml --seed 0 --run-id eval_untrained_seed0   # GPU
python -m eval.metrics runs/eval_untrained_seed0                                            # CPU is fine
```

`run_eval` writes one line per member output to `runs/<run_id>/members.jsonl`; `metrics` writes `members.csv`, `flips.csv`, `pairwise.csv` and `aggregation.csv` (95% bootstrap CIs) to `runs/<run_id>/metrics/`. Run seeds 0, 1 and 2 for every headline number. For the trained council, set `loras_from_state` in `configs/eval.yaml`. Combine seeds with:

```bash
python -m analysis.combine_seeds runs/eval_untrained_seed0 runs/eval_untrained_seed1 runs/eval_untrained_seed2 --out runs/eval_untrained_combined
```

which reports, per metric, the mean over seeds, the 95% t-interval across seeds and the widest per-seed bootstrap interval.

## Status

- [x] C1 question data
- [x] C2 pressure conditions
- [x] C3 council members (GPU check pending)
- [x] C4 protocols (GPU check pending)
- [x] C5 aggregators (ISP / OW waiting for equations from Ai et al.)
- [x] C6 rewards: CA, BTS, Pred, Calib, Flip, combined R (PTS / DMI waiting for formulas)
- [x] C7 round-robin GRPO (GPU run pending)
- [x] C8 evaluation and metrics (GPU run pending)
