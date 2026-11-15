"""The AgentTune-format exporter (research/lexsi_export.py). The AuditKIT read-back is checked by
`python research/lexsi_auditkit.py --selfcheck` in an environment that has auditkit installed."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))

import lexsi_export as lx  # noqa: E402

EVENTS = [["tier", "1: program-first"], ["model_call", 0], ["program", "print(1)"], ["output", "wrong"], ["note", "x"],
          ["model_call", 1], ["program", "print('@a[5]')"], ["output", "@a[5]"], ["final", "@a[5]"]]
PER_CALL = [{"prompt": 60, "completion": 10}, {"prompt": 40, "completion": 10}]


def test_one_step_per_model_call_with_its_action_and_result():
    steps = lx.steps_from_events(EVENTS, PER_CALL, "q")
    assert [s["step_number"] for s in steps] == [0, 1]
    assert steps[0]["action"] == {"id": "call_0_0", "name": "python", "arguments": {"code": "print(1)"}}
    assert "wrong" in steps[0]["observation"] and "[note] x" in steps[0]["observation"]
    assert steps[1]["observation"] == "@a[5]" and steps[1]["usage"] == PER_CALL[1]


def test_a_final_answer_call_is_a_step_with_no_action():
    ev = [["model_call", 0], ["call", 'view {"handle": 1}'], ["output", "rows"], ["model_call", 1], ["final", "42"]]
    steps = lx.steps_from_events(ev, [{"prompt": 1, "completion": 1}] * 2, "q")
    assert len(steps) == 2 and steps[1]["action"] is None
    assert steps[0]["action"]["name"] == "view" and steps[0]["action"]["arguments"] == {"handle": 1}


def test_retrieval_before_the_first_call_attaches_to_that_call():
    ev = [["tier", "R: document retrieval"], ["excerpts", "5 of 146 spans kept\nspan one"], ["model_call", 0], ["final", "45"]]
    steps = lx.steps_from_events(ev, [{"prompt": 9, "completion": 1}], "How long?")
    assert len(steps) == 1
    assert steps[0]["action"]["name"] == "retrieve_spans" and steps[0]["action"]["arguments"] == {"query": "How long?"}
    assert "span one" in steps[0]["observation"]


def test_pi_trace_becomes_steps_and_parallel_calls_use_tool_calls():
    trace = [{"thought": "t", "calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}},
                                        {"id": "c2", "name": "read", "arguments": "{\"path\": \"a\"}"}],
              "text": "", "usage": {"prompt": 5, "completion": 2}, "observations": ["o1", "o2"]},
             {"thought": "", "calls": [], "text": "716", "usage": {"prompt": 6, "completion": 1}, "observations": []}]
    steps = lx.steps_from_pi(trace)
    assert [c["name"] for c in steps[0]["action"]["tool_calls"]] == ["bash", "read"]
    assert steps[0]["action"]["tool_calls"][1]["arguments"] == {"path": "a"}          # string arguments are parsed
    assert steps[1]["action"] is None and steps[0]["thought"] == "t"


def test_conversation_pairs_every_call_with_its_result():
    steps = lx.steps_from_events(EVENTS, PER_CALL, "q")
    msgs = lx.conversation("q", steps, "@a[5]")
    ids = [c["id"] for m in msgs if m.get("tool_calls") for c in m["tool_calls"]]
    assert ids == [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    assert msgs[0] == {"role": "user", "content": "q"} and msgs[-1] == {"role": "assistant", "content": "@a[5]"}


def test_export_writes_the_agenttune_shape_and_provenance(tmp_path):
    rows = [{"task": "1", "final": "@a[5]", "level": "easy", "tier": 1, "seconds": 2.0, "calls": 2, "prompt_tokens": 100,
             "completion_tokens": 20, "tokens": 120, "correct": True, "events": EVENTS, "per_call": PER_CALL}]
    tasks = [{"id": "1", "prompt": "q1", "level": "easy", "labels": [["a", "5"]]}]
    assert lx.export(rows, tasks, "leanloop", "qwen3", tmp_path) == (1, 2)
    rec = json.loads((tmp_path / "trajectories.jsonl").read_text(encoding="utf-8"))
    assert {"trajectory_id", "task", "steps", "final_response", "reward", "metadata"} <= set(rec)
    assert rec["reward"] is None and rec["final_response"] == "@a[5]" and rec["metadata"]["tool_call_count"] == 2
    assert rec["metadata"]["usage"]["total_tokens"] == 120 and rec["metadata"]["conversation"]
    prov = json.loads((tmp_path / "lexsi_provenance.json").read_text(encoding="utf-8"))
    assert prov["library"] == "leanloop" and prov["training_dataset"] is None and "not by AgentTune" in prov["note"]


def test_the_runner_marks_model_calls(tmp_path):
    import ext_run
    (tmp_path / "f.csv").write_text("a\n1\n")
    chat_replies = iter([{"message": {"role": "assistant", "content": "```python\nprint('@a[1]')\n```"}, "prompt_tokens": 5, "completion_tokens": 3}])
    # run_ours builds its own Meter, so exercise the marker wrapper through the real code path with a stubbed model
    import evaluate
    original = evaluate.llm.chat
    evaluate.llm.chat = lambda *a, **k: next(chat_replies)
    try:
        r = ext_run.run_ours("lean", "Give @a. The data file is f.csv.", tmp_path, "m")
    finally:
        evaluate.llm.chat = original
    kinds = [e[0] for e in r["events"]]
    assert kinds.count("model_call") == r["calls"] == 1 and r["per_call"] == [{"prompt": 5, "completion": 3}]
    assert kinds.index("model_call") < kinds.index("program")
