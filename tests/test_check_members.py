import json

from council.check_members import check
from data.build_questions import make_record
from tests.fakes import FakeBackend, answer_json


def test_check_runs_each_member_and_writes_summary(tmp_path):
    recs = [make_record(f"q{i}", "arc_challenge", f"Q{i}?", ["a", "b", "c", "d"], "B", "mcq", "s", "dev") for i in range(4)]
    replies = {"good": lambda c: answer_json("B"), "bad": lambda c: "nope"}
    cfg = {"members": [{"name": "good", "backend": {"kind": "good"}}, {"name": "bad", "backend": {"kind": "bad"}}]}

    summary = check(cfg, recs, ["neutral", "user_wrong"], tmp_path,
                    backend_factory=lambda kind: FakeBackend(replies[kind]))

    assert summary["good"]["accuracy"] == 1.0 and summary["good"]["invalid_rate"] == 0.0
    assert summary["bad"]["over_limit"] and summary["bad"]["accuracy"] == 0.0
    lines = (tmp_path / "members.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * 4 * 2
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["good"]["n"] == 8
