"""The data split in research/safetune_guardrail.py: calibration, validation and test must never overlap."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))

import safetune_guardrail as sg  # noqa: E402


def make(tmp_path):
    with open(tmp_path / "adv.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["goal", "target"])
        w.writeheader()
        w.writerows({"goal": f"harmful request {i}", "target": "x"} for i in range(200))
    rows = [{"instruction": f"benign instruction {i}", "input": ""} for i in range(200)]
    rows += [{"instruction": "has an input so it is skipped", "input": "data"}, {"instruction": "x" * 300, "input": ""}]
    (tmp_path / "alp.json").write_text(json.dumps(rows), encoding="utf-8")
    return tmp_path / "adv.csv", tmp_path / "alp.json"


def test_splits_are_disjoint_sized_and_deterministic(tmp_path):
    adv, alp = make(tmp_path)
    h1, b1 = sg.load_prompts(adv, alp, n_test=40)
    h2, b2 = sg.load_prompts(adv, alp, n_test=40)
    assert (h1, b1) == (h2, b2)                                          # same seed, same split
    for pool in (h1, b1):
        assert {k: len(v) for k, v in pool.items()} == {"cal": 32, "val": 16, "test": 40}
        flat = pool["cal"] + pool["val"] + pool["test"]
        assert len(set(flat)) == len(flat)                               # no prompt appears in two splits
    assert all("skipped" not in p and len(p) < 200 for p in b1["cal"] + b1["val"] + b1["test"])   # only plain, short instructions


def test_the_safety_prompt_is_short_enough_to_matter_but_not_silly():
    assert 150 < len(sg.SAFETY_PROMPT) < 600
