"""Run leanloop (naive or lean mode) over the tasks. Each task gets a fresh copy of the workspace.

usage: python run_leanloop.py <naive|lean> <workspace> <tasks.json> <out.jsonl> [model]
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from leanloop.agent import run  # noqa: E402


def main(mode, ws, tasks_path, out_path, model="qwen3-8b-8k"):
    tasks = json.loads(Path(tasks_path).read_text())
    with open(out_path, "w", encoding="utf-8") as f:
        for t in tasks:
            with tempfile.TemporaryDirectory(prefix="leanloop_") as tmp:
                work = Path(tmp) / "ws"
                shutil.copytree(ws, work)
                t0 = time.time()
                res = run(t["prompt"], work, mode, model)
                res["seconds"] = round(time.time() - t0, 1)
            res["task"] = t["id"]
            res["correct"] = all(e.lower() in res["final"].lower() for e in t["expect"])
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{t['id']:18} {mode:5} steps={res['steps']:2} tokens={res['total_tokens']:6} "
                  f"{res['seconds']:6.1f}s correct={res['correct']} overflow={res['context_overflow']}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
