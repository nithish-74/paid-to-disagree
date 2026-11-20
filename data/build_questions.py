"""C1: load datasets and normalise every question to one record schema.

Schema (one JSON object per line):
    {qid, source, split, question, options, gold, answer_type, subject}

- options: list of option texts WITHOUT letter prefixes; letters are positional (A, B, ...).
- gold: a letter for answer_type "mcq", a normalised number string for "number".
- split: train | dev | test | transfer (held-out MMLU-Pro subjects).

Gold labels are stored here for evaluation only; nothing in rewards/ may read them.

Usage:
    python -m data.build_questions --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

LETTERS = "ABCDEFGHIJKLMNOP"
SPLITS = ("train", "dev", "test", "transfer")
ANSWER_TYPES = ("mcq", "number")

Record = dict[str, Any]


# ---------------------------------------------------------------- helpers


def seeded_rng(*parts: Any) -> random.Random:
    """Deterministic RNG from arbitrary parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def normalize_number(text: str) -> str:
    """'1,000' -> '1000', '3.50' -> '3.5', '$12' -> '12', '7.0' -> '7'."""
    cleaned = re.sub(r"[,$\s]", "", str(text)).rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {text!r}") from exc
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def reduce_options(
    options: list[str], gold_idx: int, n_options: int, rng: random.Random
) -> tuple[list[str], str]:
    """Keep the gold option plus (n_options-1) random distractors, then shuffle.

    Returns (options, gold_letter). Raises ValueError if there are too few options.
    """
    if len(options) < n_options:
        raise ValueError(f"need {n_options} options, got {len(options)}")
    distractor_idx = [i for i in range(len(options)) if i != gold_idx]
    chosen = [gold_idx] + rng.sample(distractor_idx, n_options - 1)
    rng.shuffle(chosen)
    return [options[i] for i in chosen], LETTERS[chosen.index(gold_idx)]


def make_record(
    qid: str,
    source: str,
    question: str,
    options: list[str],
    gold: str,
    answer_type: str,
    subject: str,
    split: str = "train",
) -> Record:
    return {
        "qid": qid,
        "source": source,
        "split": split,
        "question": question.strip(),
        "options": [o.strip() for o in options],
        "gold": gold,
        "answer_type": answer_type,
        "subject": subject,
    }


def validate_record(rec: Record) -> None:
    """Raise AssertionError if a record breaks the schema."""
    keys = {"qid", "source", "split", "question", "options", "gold", "answer_type", "subject"}
    assert set(rec) == keys, f"bad keys: {sorted(rec)}"
    assert rec["split"] in SPLITS, rec["split"]
    assert rec["answer_type"] in ANSWER_TYPES, rec["answer_type"]
    assert rec["question"], "empty question"
    if rec["answer_type"] == "mcq":
        assert len(rec["options"]) >= 2, "mcq needs options"
        assert all(rec["options"]), "empty option text"
        assert rec["gold"] in LETTERS[: len(rec["options"])], rec["gold"]
    else:
        assert rec["options"] == [], "number questions have no options"
        assert normalize_number(rec["gold"]) == rec["gold"], rec["gold"]


# ---------------------------------------------------------------- per-source normalisers
# Each takes one raw HF row and returns a record or None (item skipped).


def normalize_truthfulqa(row: dict, idx: int, n_options: int, seed: int) -> Record | None:
    targets = row["mc1_targets"]
    choices, labels = list(targets["choices"]), list(targets["labels"])
    if labels.count(1) != 1 or len(choices) < n_options:
        return None
    qid = f"tqa_{idx:04d}"
    # MC1 always lists the correct answer first, so shuffling is mandatory.
    options, gold = reduce_options(choices, labels.index(1), n_options, seeded_rng(seed, qid))
    return make_record(qid, "truthfulqa_mc1", row["question"], options, gold, "mcq", "truthfulqa")


def normalize_arc(row: dict, n_options: int, seed: int) -> Record | None:
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    if row["answerKey"] not in labels or len(texts) < n_options:
        return None
    qid = f"arc_{row['id']}"
    options, gold = reduce_options(texts, labels.index(row["answerKey"]), n_options, seeded_rng(seed, qid))
    return make_record(qid, "arc_challenge", row["question"], options, gold, "mcq", "science")


def normalize_mmlu_pro(row: dict, n_options: int, seed: int) -> Record | None:
    raw = list(row["options"])
    gold_idx = int(row["answer_index"])
    gold_text = raw[gold_idx]
    # Drop padding options and duplicates (keeping the gold one).
    options: list[str] = []
    for opt in raw:
        if opt.strip() and opt.strip().upper() != "N/A" and opt not in options:
            options.append(opt)
    if gold_text not in options or len(options) < n_options:
        return None
    qid = f"mmlupro_{row['question_id']}"
    opts, gold = reduce_options(options, options.index(gold_text), n_options, seeded_rng(seed, qid))
    return make_record(qid, "mmlu_pro", row["question"], opts, gold, "mcq", str(row["category"]))


def parse_gsm8k_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer.strip())
    if not match:
        raise ValueError("no '####' final answer")
    return normalize_number(match.group(1))


def normalize_gsm8k(row: dict, native_split: str, idx: int) -> Record | None:
    try:
        gold = parse_gsm8k_answer(row["answer"])
    except ValueError:
        return None
    qid = f"gsm8k_{native_split}_{idx:04d}"
    return make_record(qid, "gsm8k", row["question"], [], gold, "number", "math")


# ---------------------------------------------------------------- splits


def assign_splits(
    records: list[Record],
    n_dev: int,
    n_test: int,
    seed: int,
    holdout_subjects: Iterable[str] = (),
) -> list[Record]:
    """Seeded per-source split. Held-out subjects go to 'transfer' and never to train/dev/test.

    Order of priority when a source is small: test first, then dev, rest train.
    """
    holdout = set(holdout_subjects)
    by_source: dict[str, list[Record]] = {}
    for rec in records:
        by_source.setdefault(rec["source"], []).append(rec)

    out: list[Record] = []
    for source in sorted(by_source):
        pool = sorted(by_source[source], key=lambda r: r["qid"])
        transfer = [r for r in pool if r["subject"] in holdout]
        rest = [r for r in pool if r["subject"] not in holdout]
        seeded_rng(seed, "split", source).shuffle(rest)
        for i, rec in enumerate(rest):
            split = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
            out.append({**rec, "split": split})
        out.extend({**rec, "split": "transfer"} for rec in transfer)
    return out


# ---------------------------------------------------------------- loading


def _load(path: str, name: str | None, split: str):
    from datasets import load_dataset  # imported lazily so tests need no network

    return load_dataset(path, name, split=split)


def load_source(source: str, n_options: int, seed: int) -> list[Record]:
    recs: list[Record | None] = []
    if source == "truthfulqa_mc1":
        ds = _load("truthfulqa/truthful_qa", "multiple_choice", "validation")
        recs = [normalize_truthfulqa(r, i, n_options, seed) for i, r in enumerate(ds)]
    elif source == "arc_challenge":
        for split in ("train", "validation", "test"):
            recs += [normalize_arc(r, n_options, seed) for r in _load("allenai/ai2_arc", "ARC-Challenge", split)]
    elif source == "mmlu_pro":
        for split in ("validation", "test"):
            recs += [normalize_mmlu_pro(r, n_options, seed) for r in _load("TIGER-Lab/MMLU-Pro", None, split)]
    elif source == "gsm8k":
        for split in ("train", "test"):
            recs += [normalize_gsm8k(r, split, i) for i, r in enumerate(_load("openai/gsm8k", "main", split))]
    else:
        raise ValueError(f"unknown source {source!r}")
    return [r for r in recs if r is not None]


def save_jsonl(records: Iterable[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Record]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(cfg: dict, loader: Callable[[str, int, int], list[Record]] = load_source) -> list[Record]:
    out_dir = Path(cfg["out_dir"])
    all_recs: list[Record] = []
    for source in cfg["sources"]:
        raw = loader(source, cfg["n_options"], cfg["seed"])
        recs = assign_splits(raw, cfg["n_dev"], cfg["n_test"], cfg["seed"], cfg.get("holdout_subjects", []))
        for rec in recs:
            validate_record(rec)
        save_jsonl(recs, out_dir / f"{source}.jsonl")
        counts = Counter(r["split"] for r in recs)
        print(f"{source:16s} {len(recs):6d} items  " + "  ".join(f"{s}={counts.get(s, 0)}" for s in SPLITS))
        all_recs += recs
    qids = [r["qid"] for r in all_recs]
    assert len(qids) == len(set(qids)), "duplicate qids"
    return all_recs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        build(yaml.safe_load(f))


if __name__ == "__main__":
    main()
