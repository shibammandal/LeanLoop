"""FIXED (do not edit during research). Scores a harness on a task split.

The harness never reports its own tokens: every model call goes through Meter, which counts the
tokens Ollama reports and enforces the same model, context size and call cap for every harness.

Metric (accuracy first, then tokens):
  1. correct  = tasks answered right (higher is better)
  2. tokens   = total prompt + completion tokens over all tasks (lower is better)
A task also counts as wrong if it hits the call cap or overflows the context window.

usage: python research/evaluate.py <lean|lean0|naive> <dev|test> [out.jsonl]
  lean  = src/leanloop/lean.py (the file the research loop edits)
  lean0 = the original lean mode in src/leanloop/agent.py (starting point)
  naive = full-history mode in src/leanloop/agent.py (what most harnesses do)
"""

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from leanloop import llm  # noqa: E402

MODEL = "qwen3-8b-8k"
NUM_CTX = 8192
CALL_CAP = 20          # model calls per task
MAX_PREDICT = 1024     # a harness may ask for fewer output tokens, never more
WS = ROOT / "data" / "ws2"


class CapHit(Exception):
    pass


class Meter:
    def __init__(self, model=None):
        self.model = model or MODEL
        self.calls = self.prompt = self.completion = 0
        self.overflow = False
        self.log = []          # per-call usage, for trajectory export

    def __call__(self, model, messages, tools=None, **kw):
        self.calls += 1
        if self.calls > CALL_CAP:
            raise CapHit()
        r = llm.chat(self.model, messages, tools, num_ctx=NUM_CTX,
                     num_predict=min(int(kw.get("num_predict", MAX_PREDICT)), MAX_PREDICT))
        self.prompt += r["prompt_tokens"]
        self.completion += r["completion_tokens"]
        self.log.append({"prompt": r["prompt_tokens"], "completion": r["completion_tokens"]})
        self.overflow |= r["prompt_tokens"] >= NUM_CTX - 16
        return r


def check(final: str, expect: list[str]) -> bool:
    """Every expected value must appear as a whole token (so '20' does not match '200')."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", final or "")  # drop thousands separators only (1,442 -> 1442)
    return all(re.search(r"(?<![\w.])" + re.escape(e) + r"(?![\w]|\.\d)", text, re.I) for e in expect)


def load_harness(name, model=MODEL):
    if name == "lean":
        from leanloop import lean
        return lambda prompt, work, chat: lean.run(prompt, work, model, chat=chat)
    from leanloop import agent
    mode = {"lean0": "lean", "naive": "naive", "naivemap": "naivemap"}[name]
    return lambda prompt, work, chat: agent.run(prompt, work, mode, model, max_steps=CALL_CAP, chat=chat)


def main(harness, split, out_path=None):
    tasks = json.loads((WS.parent / f"{WS.name}.{split}.json").read_text())
    run = load_harness(harness)
    rows = []
    t_all = time.time()
    for t in tasks:
        meter, final, error = Meter(), "", None
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="lleval_") as tmp:
            work = Path(tmp) / "ws"
            shutil.copytree(WS, work)
            try:
                res = run(t["prompt"], work, meter)
                final = res.get("final", "") or ""
            except CapHit:
                error = "call cap"
            except Exception as e:  # a crashing harness scores the task as wrong
                error = f"{type(e).__name__}: {e}"
        ok = check(final, t["expect"]) and error is None and not meter.overflow
        rows.append({"task": t["id"], "correct": ok, "calls": meter.calls, "prompt_tokens": meter.prompt,
                     "completion_tokens": meter.completion, "tokens": meter.prompt + meter.completion,
                     "seconds": round(time.time() - t0, 1), "overflow": meter.overflow, "error": error,
                     "final": final[:300]})
        r = rows[-1]
        print(f"  {r['task']:20} {'ok' if ok else '--'} calls={r['calls']:2} tokens={r['tokens']:6} "
              f"{r['seconds']:5.1f}s{'  ' + error if error else ''}", flush=True)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    correct = sum(r["correct"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)
    print("---")
    print(f"correct:  {correct}/{len(rows)}")
    print(f"tokens:   {tokens}")
    print(f"calls:    {sum(r['calls'] for r in rows)}")
    print(f"seconds:  {round(time.time() - t_all, 1)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
