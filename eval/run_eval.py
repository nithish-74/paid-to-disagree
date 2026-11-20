"""C8: run every protocol x condition x dataset cell on the test split.

Writes to runs/<run_id>/:
    members.jsonl    one line per member output (every stage; Karpathy rankings too)
    questions.jsonl  the exact test questions used (so metrics never re-derive them)
    config.json      the resolved eval config
Aggregators are applied afterwards by eval/metrics.py, so every aggregator sees exactly
the same member outputs. No metric is computed here.

    python -m eval.run_eval --config configs/eval.yaml
    python -m eval.run_eval --config configs/eval.yaml --seed 1 --run-id eval_untrained_seed1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from council.members import Backend, CouncilMember, VLLMBackend
from council.protocols import ProtocolResult, run_protocol
from data.build_questions import Record, load_jsonl, save_jsonl
from data.pressure import apply_condition


def resolve_loras(cfg: dict[str, Any]) -> dict[str, str]:
    loras = dict(cfg.get("loras") or {})
    state_path = cfg.get("loras_from_state")
    if state_path:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        loras = {**state.get("loras", {}), **loras}
    return loras


def build_members(cfg: dict[str, Any], backend_factory: Callable[..., Backend] = VLLMBackend) -> list[CouncilMember]:
    with open(cfg["council_config"], encoding="utf-8") as f:
        council = yaml.safe_load(f)
    gen = council.get("generation", {})
    loras = resolve_loras(cfg)
    members = []
    for m in council["members"]:
        kwargs = {**m["backend"], "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.28)}
        if m["name"] in loras:
            kwargs.update(lora_path=loras[m["name"]], max_lora_rank=cfg.get("max_lora_rank", 8))
        members.append(CouncilMember(
            name=m["name"],
            backend=backend_factory(**kwargs),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=gen.get("max_tokens", 256),
            seed=cfg["seed"],
            token_prob_fallback=gen.get("token_prob_fallback", True),
            always_option_probs=gen.get("always_option_probs", False),
        ))
    return members


def flatten(result: ProtocolResult, dataset: str, pressure_target: str | None, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """One line per output in every stage, tagged with its cell."""
    lines = []
    for stage, outputs in result.stages.items():
        for o in outputs:
            lines.append({
                **meta,
                "dataset": dataset,
                "protocol": result.protocol,
                "condition": result.condition,
                "qid": result.qid,
                "pressure_target": pressure_target,
                **{k: v for k, v in o.items() if k not in ("qid", "condition", "stage")},
                "stage": stage,
                "member_stage": o.get("stage"),
                "protocol_meta": result.meta or None,
            })
    return lines


def evaluate_dataset(
    records: list[Record],
    members: list[CouncilMember],
    protocols: list[dict[str, Any]],
    conditions: list[str],
    seed: int,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    recs = {r["qid"]: r for r in records}
    dataset = records[0]["source"] if records else "unknown"
    prompts = [apply_condition(r, c, seed) for r in records for c in conditions]
    targets = {(p.qid, p.condition): p.pressure_target for p in prompts}
    lines: list[dict[str, Any]] = []
    for proto in protocols:
        kwargs = {k: v for k, v in proto.items() if k != "name"}
        if proto["name"] == "karpathy_style":
            kwargs.setdefault("seed", seed)
        start = time.time()
        for res in run_protocol(proto["name"], prompts, recs, members, **kwargs):
            lines += flatten(res, dataset, targets[(res.qid, res.condition)], meta)
        print(f"  {dataset:16s} {proto['name']:15s} {len(prompts)} prompts  {time.time() - start:.0f}s")
    return lines


def run(cfg: dict[str, Any], out_dir: Path, members: list[CouncilMember]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**cfg, "loras_resolved": resolve_loras(cfg)}, indent=2), encoding="utf-8")
    meta = {"run_id": cfg["run_id"], "seed": cfg["seed"]}
    members_path = out_dir / "members.jsonl"
    members_path.unlink(missing_ok=True)
    all_questions: list[Record] = []
    n_lines = 0
    for path in cfg["datasets"]:
        records = [r for r in load_jsonl(Path(path)) if r["split"] == cfg.get("split", "test")]
        if cfg.get("n"):
            records = records[: cfg["n"]]
        if not records:
            continue
        all_questions += records
        lines = evaluate_dataset(records, members, cfg["protocols"], cfg["conditions"], cfg["seed"], meta)
        with members_path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        n_lines += len(lines)
    save_jsonl(all_questions, out_dir / "questions.jsonl")
    (out_dir / "member_stats.json").write_text(
        json.dumps({m.name: m.stats.summary() for m in members}, indent=2), encoding="utf-8")
    return n_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="C8 evaluation run (GPU)")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_id:
        cfg["run_id"] = args.run_id
    out_dir = Path("runs") / cfg["run_id"]
    n = run(cfg, out_dir, build_members(cfg))
    print(f"wrote {n} lines -> {out_dir / 'members.jsonl'}; next: python -m eval.metrics {out_dir}")


if __name__ == "__main__":
    main()
