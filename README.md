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

## Status

- [x] C1 question data
- [x] C2 pressure conditions
- [ ] C3 council members
- [ ] C4 protocols
- [ ] C5 aggregators
- [ ] C6 rewards
- [ ] C7 training
- [ ] C8 evaluation
