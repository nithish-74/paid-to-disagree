"""Phase 1 exit check: every member answers the dev set with < 5% invalid JSON.

Loads ONE member at a time (fits a single 16 GB T4), runs it on the dev split under the
chosen pressure conditions, writes every output to runs/<run_id>/members.jsonl and prints
a per-member summary (invalid-JSON rate, peer_prediction fallback rate, dev accuracy).

Accuracy uses dev gold labels for monitoring only; nothing here feeds a reward.

Kaggle:
    python -m council.check_members --data data/processed/arc_challenge.jsonl --n 50
    python -m council.check_members --data data/processed/truthfulqa_mc1.jsonl --conditions neutral user_wrong
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import yaml

from council.members import Backend, CouncilMember, load_members, make_backend, write_outputs
from data.build_questions import load_jsonl
from data.pressure import CONDITIONS, apply_condition


def check(
    cfg: dict,
    records: list[dict],
    conditions: list[str],
    run_dir: Path,
    backend_factory: Callable[..., Backend] = make_backend,
) -> dict[str, dict]:
    recs = {r["qid"]: r for r in records}
    prompts = [apply_condition(r, c) for r in records for c in conditions]
    summary: dict[str, dict] = {}
    for member_cfg in cfg["members"]:
        [member] = load_members({**cfg, "members": [member_cfg]}, backend_factory)
        start = time.time()
        turns = member.answer_prompts(prompts, recs)
        outputs = [o for t in turns for o in t]
        write_outputs(outputs, run_dir / "members.jsonl")
        finals = [t[-1] for t in turns]
        correct = sum(o.valid and o.answer == recs[o.qid]["gold"] for o in finals)
        summary[member.name] = {
            **member.stats.summary(),
            "accuracy": round(correct / len(finals), 4) if finals else 0.0,
            "seconds": round(time.time() - start, 1),
        }
        _close(member)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _close(member: CouncilMember) -> None:
    close = getattr(member.backend, "close", None)
    if callable(close):
        close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 member check (invalid-JSON rate < 5%).")
    parser.add_argument("--config", default="configs/council.yaml")
    parser.add_argument("--data", required=True, help="processed JSONL from data/build_questions.py")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--n", type=int, default=0, help="limit to first n questions (0 = all)")
    parser.add_argument("--conditions", nargs="+", default=["neutral"], choices=CONDITIONS)
    parser.add_argument("--run-id", default=time.strftime("check_%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    records = [r for r in load_jsonl(Path(args.data)) if r["split"] == args.split]
    if args.n:
        records = records[: args.n]
    run_dir = Path("runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "council": cfg}, indent=2), encoding="utf-8")

    summary = check(cfg, records, args.conditions, run_dir)
    print(f"\n{len(records)} questions x {len(args.conditions)} conditions -> {run_dir}")
    for name, s in summary.items():
        flag = "  <-- FIX PROMPT (>5% invalid)" if s["over_limit"] else ""
        print(f"{name:8s} invalid={s['invalid_rate']:.1%} (first try {s['first_try_invalid_rate']:.1%})  "
              f"pp_fallback={s['pp_fallback_rate']:.1%}  acc={s['accuracy']:.1%}  {s['seconds']}s{flag}")


if __name__ == "__main__":
    main()
