from leanloop.agent import KEEP_FULL, run
from leanloop.store import Store


def script(steps):
    """Fake chat(): replays scripted messages; records the prompts it was sent."""
    seen = []
    it = iter(steps)

    def chat(model, messages, tools, **kw):
        seen.append([dict(m) for m in messages])
        msg = next(it)
        return {"message": msg, "prompt_tokens": sum(len(str(m.get("content", ""))) for m in messages) // 4,
                "completion_tokens": 5}
    return chat, seen


def call(name, **a):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": a}}]}


def test_store_fetch_and_search():
    s = Store()
    h = s.put("\n".join(f"line {i}" for i in range(1, 101)))
    assert "5: line 5" in s.fetch(h, 5, 6) and "[lines 5-6 of 100]" in s.fetch(h, 5, 6)
    assert "[3 matching lines" in s.search(h, r"line (7|8|9)$")
    assert s.fetch(99).startswith("error")


def test_lean_pages_big_output_and_answers_via_fetch(tmp_path):
    (tmp_path / "big.txt").write_text("\n".join(f"row {i} value {i * 3}" for i in range(500)))
    chat, seen = script([call("read_file", path="big.txt"), call("search", handle=1, pattern="row 42 "),
                         {"role": "assistant", "content": "126"}])
    r = run("q", tmp_path, "lean", "m", chat=chat)
    assert r["final"] == "126" and r["stopped"] == "answer"
    second_prompt = " ".join(m.get("content", "") for m in seen[1])
    assert "[#1:" in second_prompt and "row 400 value" not in second_prompt  # receipt, not the whole file
    assert "row 42 value 126" in " ".join(m.get("content", "") for m in seen[2])  # search result reached the model


def test_naive_keeps_everything(tmp_path):
    (tmp_path / "f.txt").write_text("x" * 3000)
    chat, seen = script([call("read_file", path="f.txt"), {"role": "assistant", "content": "done"}])
    run("q", tmp_path, "naive", "m", chat=chat)
    assert "x" * 3000 in " ".join(m.get("content", "") for m in seen[1])


def test_lean_evicts_old_outputs_and_dedupes(tmp_path):
    for i in range(4):
        (tmp_path / f"f{i}.txt").write_text(f"content-{i} " * 10)
    steps = [call("read_file", path=f"f{i}.txt") for i in range(4)] + [call("read_file", path="f0.txt"),
                                                                       {"role": "assistant", "content": "ok"}]
    chat, seen = script(steps)
    r = run("q", tmp_path, "lean", "m", chat=chat)
    tool_msgs = [m["content"] for m in seen[4] if m["role"] == "tool"]
    assert sum("evicted" in c for c in tool_msgs) == 4 - KEEP_FULL
    assert "duplicate of step 1" in " ".join(m.get("content", "") for m in seen[5] if m["role"] == "tool")
    assert r["steps"] == 6


def test_budget_stop(tmp_path):
    chat, _ = script([call("bash", command=f"echo {i}") for i in range(3)])
    r = run("q", tmp_path, "lean", "m", max_steps=3, chat=chat)
    assert r["stopped"] == "budget" and r["steps"] == 3
