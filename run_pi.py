"""Run pi (unmodified) over the same tasks through Ollama and total its token usage.

usage: python run_pi.py <workspace> <tasks.json> <out.jsonl> [model]
Pi runs with `--mode json --no-session --thinking off`, in a throwaway copy of the workspace.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "research"))
from evaluate import check  # noqa: E402  (same scorer as the other harnesses)

ROOT = Path(__file__).parent
PI_CLI = ROOT / "tools" / "pi" / "node_modules" / "@earendil-works" / "pi-coding-agent" / "dist" / "bundle" / "cli.js"


def _cap(x, n=1500):
    return x if not isinstance(x, str) or len(x) <= n else x[:n] + " ...[cut]"


def parse(stdout: str) -> dict:
    prompt = completion = steps = 0
    final, per_call = "", []
    trace, by_call = [], {}     # step-by-step: thought, tool calls, their results, token usage
    for line in stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = e.get("message") or {}
        if e.get("type") == "message_end" and m.get("role") == "assistant":
            u = m.get("usage", {})
            p = u.get("input", 0) + u.get("cacheRead", 0) + u.get("cacheWrite", 0)
            prompt, completion, steps = prompt + p, completion + u.get("output", 0), steps + 1
            per_call.append({"step": steps, "prompt": p, "completion": u.get("output", 0)})
            text = "".join(c.get("text", "") for c in m.get("content", []) if c.get("type") == "text")
            if text:
                final = text
            parts = m.get("content", [])
            step = {"thought": _cap("".join(c.get("thinking", "") for c in parts if c.get("type") == "thinking"), 600),
                    "calls": [{"id": c.get("id"), "name": c.get("name"), "arguments": c.get("arguments")}
                              for c in parts if c.get("type") == "toolCall"],
                    "text": _cap(text), "usage": {"prompt": p, "completion": u.get("output", 0)}, "observations": []}
            trace.append(step)
            for c in step["calls"]:
                by_call[c["id"]] = step
        elif e.get("type") == "message_end" and m.get("role") == "toolResult":
            step = by_call.get(m.get("toolCallId"))
            if step is not None:
                step["observations"].append(_cap("".join(c.get("text", "") for c in m.get("content", []) if c.get("type") == "text")))
    return {"mode": "pi", "final": final, "steps": steps, "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, "per_call": per_call, "trace": trace}


def main(ws, tasks_path, out_path, model="qwen3-8b-8k"):
    tasks = json.loads(Path(tasks_path).read_text())
    env = dict(os.environ, PI_CODING_AGENT_DIR=str(ROOT / "pi-agent"))
    with open(out_path, "w", encoding="utf-8") as f:
        for t in tasks:
            with tempfile.TemporaryDirectory(prefix="pi_") as tmp:
                work = Path(tmp) / "ws"
                shutil.copytree(ws, work)
                t0 = time.time()
                try:
                    r = subprocess.run(["node", str(PI_CLI), "--provider", "ollama", "--model", model, "--no-session",
                                        "--thinking", "off", "--mode", "json", t["prompt"]], cwd=work, env=env,
                                       stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
                    res = parse(r.stdout)
                except subprocess.TimeoutExpired:
                    res = {"mode": "pi", "final": "", "steps": 0, "prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0, "per_call": [], "timeout": True}
            res["seconds"] = round(time.time() - t0, 1)
            res["task"] = t["id"]
            res["correct"] = check(res["final"], t["expect"])
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{t['id']:18} pi    steps={res['steps']:2} tokens={res['total_tokens']:6} {res['seconds']:6.1f}s correct={res['correct']}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
