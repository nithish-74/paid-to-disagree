"""C1: load datasets and normalise every question to one record schema.

Schema (one JSON object per line):
    {qid, source, split, question, options, gold, answer_type, subject}

- options: list of option texts WITHOUT letter prefixes; letters are positional (A, B, ...).
- gold: a letter for answer_type "mcq", a normalised number string for "number".
- split: train | dev | test | transfer (held-out MMLU-Pro subjects).
  Evaluation-only sources (SycophancyEval, SYCON-Bench closed-form, GPQA-diamond) are all "test".

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


# ---------------------------------------------------------------- evaluation-only sources
# These sets are used for evaluation only: every record goes to split "test" (see EVAL_ONLY_SOURCES).
# Some items have only two answer choices; those stay 2-option MCQs (chance = 50%) instead of
# being padded to n_options (user decision, 2026-11-26).

EVAL_ONLY_SOURCES = ("sycophancyeval_answer", "sycophancyeval_are_you_sure", "sycon_fp", "gpqa_diamond")

# Pinned commits so a rebuild always sees the same files.
SYCOPHANCY_EVAL_URL = (
    "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/"
    "9a1694221e3639887138f61deae344335eca6752/datasets/{name}.jsonl"
)
SYCON_FP_URL = (
    "https://raw.githubusercontent.com/JiseungHong/SYCON-Bench/"
    "6356b923905ff5b05877ac7bb65e5207be00e1c2/false-presuppositions-setting/data/{name}.txt"
)
SYCON_FP_PROMPT = "{question}\nWhich statement is correct?"
AQUA_OPTION = re.compile(r"^\(([A-E])\)(.*)$")


def question_key(text: str) -> str:
    """Case- and whitespace-insensitive key used to find the same question in two datasets."""
    return " ".join(str(text).lower().split())


def _text_id(text: str) -> str:
    return hashlib.sha256(question_key(text).encode("utf-8")).hexdigest()[:12]


def _mcq_from_texts(
    qid: str, source: str, question: str, texts: list, gold_idx: int, subject: str, n_options: int, seed: int
) -> Record | None:
    """Clean option texts, drop duplicates (keeping the gold one), then reduce to at most n_options.

    Returns None if an option is not a non-empty string, if the gold text is duplicated
    (ambiguous item), or if fewer than two distinct options remain.
    """
    if not all(isinstance(t, str) and t.strip() for t in texts):
        return None
    clean = [t.strip() for t in texts]
    gold_text = clean[gold_idx]
    if clean.count(gold_text) > 1:
        return None
    options = list(dict.fromkeys(clean))
    k = min(n_options, len(options))
    if k < 2:
        return None
    opts, gold = reduce_options(options, options.index(gold_text), k, seeded_rng(seed, qid))
    return make_record(qid, source, question, opts, gold, "mcq", subject)


def normalize_sycophancy_answer(row: dict, n_options: int, seed: int) -> Record | None:
    """SycophancyEval answer.jsonl. Only the trivia_qa part is used, as a 2-option MCQ
    (correct_answer vs incorrect_answer); its truthful_qa part repeats our TruthfulQA questions.

    The dataset's own suggestion text is not kept: C2 adds the pressure conditions.
    """
    base = row["base"]
    if base.get("dataset") != "trivia_qa":
        return None
    correct, wrong = base.get("correct_answer"), base.get("incorrect_answer")
    if not isinstance(correct, str) or not isinstance(wrong, str):
        return None
    aliases = {question_key(a) for a in base.get("answer", [])} | {question_key(correct)}
    if question_key(wrong) in aliases:  # the "incorrect" answer is an accepted alias
        return None
    qid = f"se_answer_trivia_{_text_id(base['question'])}"
    return _mcq_from_texts(
        qid, "sycophancyeval_answer", base["question"], [correct, wrong], 0, "trivia_qa", n_options, seed
    )


def _parse_aqua_answers(answers: str) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for line in answers.strip().split("\n"):
        match = AQUA_OPTION.match(line.strip())
        if not match:
            return None
        out[match.group(1)] = match.group(2)
    return out


def normalize_sycophancy_are_you_sure(row: dict, n_options: int, seed: int) -> Record | None:
    """SycophancyEval are_you_sure.jsonl, closed-form parts only:
    mmlu_mc_cot (4 options), aqua_mc (5 -> n_options), math_mc_cot (2 options).

    truthful_qa_mc repeats our TruthfulQA questions; trivia_qa / truthful_qa are free-form. All skipped.
    """
    base = row["base"]
    ds = base.get("dataset")
    qid = f"se_ays_{ds}_{_text_id(base['question'])}"
    source = "sycophancyeval_are_you_sure"
    if ds == "mmlu_mc_cot":
        texts = [base.get(letter) for letter in "ABCD"]
        gold_idx = "ABCD".index(base["correct_letter"])
        return _mcq_from_texts(qid, source, base["question"], texts, gold_idx, "mmlu", n_options, seed)
    if ds == "aqua_mc":
        parsed = _parse_aqua_answers(base["answers"])
        if parsed is None or base["correct_letter"] not in parsed:
            return None
        letters = sorted(parsed)
        texts = [parsed[letter] for letter in letters]
        gold_idx = letters.index(base["correct_letter"])
        return _mcq_from_texts(qid, source, base["question"], texts, gold_idx, "aqua", n_options, seed)
    if ds == "math_mc_cot":
        texts = [base.get("correct_answer"), base.get("wrong_answer")]
        return _mcq_from_texts(qid, source, base["question"], texts, 0, "math", n_options, seed)
    return None


def normalize_sycon_fp(question: str, presupposition: str, correction: str, idx: int, seed: int) -> Record | None:
    """SYCON-Bench false-presupposition item -> closed-form 2-option MCQ (user-approved design).

    Options are the false presupposition and its correction in seeded order; gold = the correction.
    """
    question, presupposition, correction = question.strip(), presupposition.strip(), correction.strip()
    if not (question and presupposition and correction) or question_key(presupposition) == question_key(correction):
        return None
    qid = f"sycon_fp_{idx:03d}"
    options, gold = reduce_options([correction, presupposition], 0, 2, seeded_rng(seed, qid))
    prompt = SYCON_FP_PROMPT.format(question=question)
    return make_record(qid, "sycon_fp", prompt, options, gold, "mcq", "false_presupposition")


def normalize_gpqa(row: dict, idx: int, n_options: int, seed: int) -> Record | None:
    """GPQA (gated HF dataset Idavidrein/gpqa). Column names follow the dataset card;
    not yet checked against the real file (needs gated access)."""
    texts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    record_id = str(row.get("Record ID") or f"{idx:04d}").strip()
    qid = f"gpqa_{record_id}"
    subject = str(row.get("Subdomain") or row.get("High-level domain") or "gpqa").strip()
    return _mcq_from_texts(qid, "gpqa_diamond", row["Question"], texts, 0, subject, n_options, seed)


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


def assign_eval_only(records: list[Record]) -> list[Record]:
    """Evaluation-only sources: every record goes to 'test', sorted by qid."""
    return [{**rec, "split": "test"} for rec in sorted(records, key=lambda r: r["qid"])]


def dedupe_questions(records: list[Record]) -> list[Record]:
    """Keep the first record per question text (SycophancyEval repeats questions across templates)."""
    out: dict[str, Record] = {}
    for rec in records:
        out.setdefault(question_key(rec["question"]), rec)
    return list(out.values())


def drop_overlap(records: list[Record], reference: Iterable[Record]) -> list[Record]:
    """Remove records whose question also appears (case/space-insensitive) in `reference`,
    so an evaluation-only set never repeats a question from a training source."""
    seen = {question_key(r["question"]) for r in reference}
    return [r for r in records if question_key(r["question"]) not in seen]


# ---------------------------------------------------------------- loading


def _load(path: str, name: str | None, split: str):
    from datasets import load_dataset  # imported lazily so tests need no network

    return load_dataset(path, name, split=split)


def _fetch_text(url: str) -> str:
    from urllib.request import urlopen

    with urlopen(url, timeout=120) as resp:
        return resp.read().decode("utf-8")


def _fetch_jsonl(url: str) -> list[dict]:
    return [json.loads(line) for line in _fetch_text(url).splitlines() if line.strip()]


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
    elif source == "sycophancyeval_answer":
        rows = _fetch_jsonl(SYCOPHANCY_EVAL_URL.format(name="answer"))
        recs = [normalize_sycophancy_answer(r, n_options, seed) for r in rows]
    elif source == "sycophancyeval_are_you_sure":
        rows = _fetch_jsonl(SYCOPHANCY_EVAL_URL.format(name="are_you_sure"))
        recs = [normalize_sycophancy_are_you_sure(r, n_options, seed) for r in rows]
    elif source == "sycon_fp":
        questions, presups, corrections = (
            _fetch_text(SYCON_FP_URL.format(name=n)).splitlines()
            for n in ("questions", "presuppositions", "corrections")
        )
        if not len(questions) == len(presups) == len(corrections):
            raise ValueError("SYCON-Bench false-presupposition files are not line-aligned")
        recs = [normalize_sycon_fp(*item, i, seed) for i, item in enumerate(zip(questions, presups, corrections))]
    elif source == "gpqa_diamond":
        # Gated: accept the terms on huggingface.co/datasets/Idavidrein/gpqa and log in first.
        ds = _load("Idavidrein/gpqa", "gpqa_diamond", "train")
        recs = [normalize_gpqa(r, i, n_options, seed) for i, r in enumerate(ds)]
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


def _overlap_reference(cfg: dict, built: dict[str, list[Record]]) -> list[Record]:
    """Records of the training sources named in cfg['overlap_reference'], from this run or from disk."""
    out: list[Record] = []
    for name in cfg.get("overlap_reference", []):
        if name in built:
            out += built[name]
            continue
        path = Path(cfg["out_dir"]) / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} is needed to drop overlapping questions; build {name!r} first")
        out += load_jsonl(path)
    return out


def build(
    cfg: dict,
    loader: Callable[[str, int, int], list[Record]] = load_source,
    sources: Iterable[str] | None = None,
) -> list[Record]:
    """Build each source into out_dir/<source>.jsonl.

    Training sources get assign_splits. Evaluation-only sources (EVAL_ONLY_SOURCES) are de-duplicated,
    lose any question that appears in a cfg['overlap_reference'] source, and go entirely to 'test'.
    """
    out_dir = Path(cfg["out_dir"])
    names = list(cfg["sources"] if sources is None else sources)
    # Training sources first, so evaluation-only sources can be checked against them.
    names.sort(key=lambda s: s in EVAL_ONLY_SOURCES)
    built: dict[str, list[Record]] = {}
    for source in names:
        raw = loader(source, cfg["n_options"], cfg["seed"])
        if source in EVAL_ONLY_SOURCES:
            kept = drop_overlap(dedupe_questions(raw), _overlap_reference(cfg, built))
            recs = assign_eval_only(kept)
        else:
            recs = assign_splits(raw, cfg["n_dev"], cfg["n_test"], cfg["seed"], cfg.get("holdout_subjects", []))
        for rec in recs:
            validate_record(rec)
        save_jsonl(recs, out_dir / f"{source}.jsonl")
        counts = Counter(r["split"] for r in recs)
        print(f"{source:28s} {len(recs):6d} items  " + "  ".join(f"{s}={counts.get(s, 0)}" for s in SPLITS))
        built[source] = recs
    all_recs = [rec for recs in built.values() for rec in recs]
    qids = [r["qid"] for r in all_recs]
    assert len(qids) == len(set(qids)), "duplicate qids"
    return all_recs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--sources", nargs="+", help="build only these sources (default: config 'sources')")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        build(yaml.safe_load(f), sources=args.sources)


if __name__ == "__main__":
    main()
