"""B8: synthetic-data anti-sycophancy SFT (Wei et al. 2023, arXiv 2308.03958), adapted to the council.

Per member, two GPU commands (one Kaggle session can run both for Qwen/Llama; Gemma is slower):

    # 1. answer the synthetic claims WITHOUT the user's opinion (filtration, Appendix C.4) and the
    #    member's own training questions (the instruction-data mix, Section 4.2), temperature 0
    python -m train.b8_sft filter --config configs/b8.yaml --member qwen
    # 2. LoRA fine-tune on: opinion prompt -> the member's own no-opinion answer, mixed 5:1
    python -m train.b8_sft train  --config configs/b8.yaml --member qwen

Approved adaptation (paper -> here):
- target text: the paper's "I believe the best answer is (A)" -> the member's own JSON answer to the
  same claim with the opinion removed. That is the paper's rule (the claim's truth does not depend
  on the user's opinion) in the council's output format, with no hand-written text.
- filtration: as the paper, keep only claims the member answers correctly without the opinion.
- instruction mix (paper: Flan at 5:1) -> the member's own valid answers to C1 training questions
  (neutral condition), 5 synthetic : 1 self example. Label-free: any valid answer is kept.
- fine-tuning: LoRA r=8 (as our GRPO) instead of full fine-tuning; 1k steps and batch 32 as the
  paper; lr 1e-4 (the paper's 3e-3 is for full Adafactor tuning of PaLM).
Targets need a model-written peer_prediction (no token-probability fallback), so every target is
text the member itself produced.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from data.build_questions import Record, load_jsonl, save_jsonl, seeded_rng
from data.pressure import apply_condition

Output = dict[str, Any]


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sft_target(out: Output) -> str | None:
    """The member's own answer as canonical JSON (the key order of the council prompt), or None
    when it is not usable as a target: invalid, or peer_prediction not written by the model."""
    if not out.get("valid") or out.get("answer") is None or out.get("peer_prediction_source") != "model":
        return None
    return json.dumps({"reason": out.get("reason", ""), "answer": out["answer"],
                       "confidence": out.get("confidence"), "peer_prediction": out.get("peer_prediction")},
                      ensure_ascii=False)


def prompt_messages(rec: Record) -> list[dict[str, str]]:
    """System prompt + the neutral user turn, exactly as the council member sends them."""
    from council.members import system_prompt

    return [{"role": "system", "content": system_prompt(rec)}] + apply_condition(rec, "neutral").messages


def synthetic_examples(records: list[Record], plain_outputs: dict[str, Output]) -> list[dict[str, Any]]:
    """Filtration + targets: keep a synthetic item only if the member's no-opinion answer is right
    (and usable); the example is the WITH-opinion prompt -> that no-opinion answer."""
    out = []
    for rec in records:
        o = plain_outputs.get(rec["qid"])
        target = sft_target(o) if o else None
        if target is None or o["answer"] != rec["gold"]:
            continue
        out.append({"prompt": prompt_messages(rec), "completion": [{"role": "assistant", "content": target}],
                    "qid": rec["qid"], "kind": "synthetic"})
    return out


def self_examples(records: list[Record], outputs: dict[str, Output]) -> list[dict[str, Any]]:
    """Instruction-data stand-in: the member's own valid neutral answers to C1 training questions."""
    out = []
    for rec in records:
        o = outputs.get(rec["qid"])
        target = sft_target(o) if o else None
        if target is not None:
            out.append({"prompt": prompt_messages(rec), "completion": [{"role": "assistant", "content": target}],
                        "qid": rec["qid"], "kind": "self"})
    return out


def mix(synthetic: list[dict], selfdata: list[dict], ratio: int, seed: int) -> list[dict]:
    """ratio synthetic : 1 self (paper 5:1). Self examples are sampled without replacement when there
    are enough, else with replacement; the result is shuffled with the seed."""
    rng = random.Random(seed)
    n_self = len(synthetic) // ratio
    if not selfdata or n_self == 0:
        chosen: list[dict] = []
    elif n_self <= len(selfdata):
        chosen = rng.sample(selfdata, n_self)
    else:
        chosen = [rng.choice(selfdata) for _ in range(n_self)]
    rows = synthetic + chosen
    rng.shuffle(rows)
    return rows


def out_dir(cfg: dict[str, Any]) -> Path:
    return Path("runs") / cfg["experiment"]


def self_records(cfg: dict[str, Any]) -> list[Record]:
    recs = [r for p in cfg["self_data"]["files"] for r in load_jsonl(Path(p)) if r["split"] == "train"]
    recs.sort(key=lambda r: r["qid"])
    seeded_rng(cfg["seed"], "b8-self").shuffle(recs)
    return recs[: cfg["self_data"]["n_questions"]]


# ================================================================ GPU commands


def cmd_filter(cfg: dict[str, Any], member_name: str) -> None:
    from council.members import CouncilMember, make_backend
    from data.b8_synthetic import plain_record
    from train.grpo_roundrobin import member_cfg

    out = out_dir(cfg)
    synth_path = out / "synthetic.jsonl"
    synth = load_jsonl(synth_path)
    n = cfg["n_prompts"].get(member_name, cfg["n_prompts"]["default"])
    synth = synth[:n]
    council = load_yaml(cfg["council_config"])
    backend = make_backend(**member_cfg(council, member_name)["backend"])
    member = CouncilMember(member_name, backend, temperature=0.0, max_tokens=256, token_prob_fallback=False)

    def answer(records: list[Record]) -> dict[str, Output]:
        prompts = [apply_condition(r, "neutral") for r in records]
        recs = {r["qid"]: r for r in records}
        return {o.qid: o.to_dict() for turns in member.answer_prompts(prompts, recs) for o in turns}

    plain = [plain_record(r) for r in synth]
    plain_out = answer(plain)
    selfrecs = self_records(cfg)
    self_out = answer(selfrecs)
    backend.close()

    syn_ex = synthetic_examples(synth, plain_out)
    self_ex = self_examples(selfrecs, self_out)
    acc = sum(o.get("answer") == r["gold"] for r, o in ((r, plain_out.get(r["qid"], {})) for r in plain)) / max(1, len(plain))
    stats = {"member": member_name, "n_prompts": len(synth), "plain_accuracy": round(acc, 4),
             "kept_synthetic": len(syn_ex), "self_examples": len(self_ex), "member_stats": member.stats.summary()}
    save_jsonl(syn_ex, out / f"{member_name}.synthetic_sft.jsonl")
    save_jsonl(self_ex, out / f"{member_name}.self_sft.jsonl")
    (out / f"{member_name}.filter_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("filter:", stats)


def cmd_train(cfg: dict[str, Any], member_name: str) -> None:
    from unsloth import FastLanguageModel  # noqa: I001  (must come before trl / transformers)
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from train.grpo_roundrobin import member_cfg, train_settings

    out = out_dir(cfg)
    s = cfg["sft"]
    rows = mix(load_jsonl(out / f"{member_name}.synthetic_sft.jsonl"),
               load_jsonl(out / f"{member_name}.self_sft.jsonl"), s["mix_ratio"], cfg["seed"])
    rows = [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    mc = member_cfg(load_yaml(cfg["council_config"]), member_name)
    ts = train_settings(mc)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mc["backend"]["model"], max_seq_length=s["max_seq_length"], load_in_4bit=False,
        dtype=getattr(torch, ts["dtype"]))
    model = FastLanguageModel.get_peft_model(
        model, r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth", random_state=cfg["seed"])
    leg = out / member_name
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=Dataset.from_list(rows),
        args=SFTConfig(output_dir=str(leg / "trainer"), max_steps=s["steps"], per_device_train_batch_size=s["batch_size"],
                       gradient_accumulation_steps=s["gradient_accumulation_steps"], learning_rate=s["learning_rate"],
                       logging_steps=10, save_strategy="no", fp16=ts["fp16"], bf16=ts["bf16"], seed=cfg["seed"],
                       max_length=s["max_seq_length"], completion_only_loss=True, report_to="none"))
    trainer.train()
    path = leg / "final"
    model.save_lora(str(path)) if hasattr(model, "save_lora") else model.save_pretrained(str(path))
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"loras": {}}
    state["loras"][member_name] = str(path)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"done: {member_name} -> {path} ({len(rows)} rows)")


def cmd_generate(cfg: dict[str, Any]) -> None:
    """CPU + network: build the synthetic prompts once (shared by all members)."""
    from data.b8_synthetic import generate, load_all_pairs

    n = max(v for v in cfg["n_prompts"].values())
    recs = generate(load_all_pairs(), n, cfg["seed"])
    save_jsonl(recs, out_dir(cfg) / "synthetic.jsonl")
    print(f"generated {len(recs)} synthetic prompts -> {out_dir(cfg) / 'synthetic.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="B8: synthetic-data anti-sycophancy SFT (Wei et al. 2023)")
    parser.add_argument("command", choices=["generate", "filter", "train"])
    parser.add_argument("--config", default="configs/b8.yaml")
    parser.add_argument("--member")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.command == "generate":
        cmd_generate(cfg)
    elif not args.member:
        parser.error("--member is required for filter / train")
    elif args.command == "filter":
        cmd_filter(cfg, args.member)
    else:
        cmd_train(cfg, args.member)


if __name__ == "__main__":
    main()
