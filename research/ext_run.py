"""Run a harness on an EXTERNAL benchmark suite (built by research/prepare_dabench.py) and score it.

Every harness gets the same thing: one throwaway workspace per question and the same prompt. Tokens come from Ollama's own
counts (via Meter for leanloop/naive, via pi's JSON usage events for pi). A missing or malformed answer counts as WRONG.

DABench scoring mirrors the benchmark's official closed-form rule: tags `@name[value]` are pulled out with
`@(\\w+)\\[(.*?)\\]`; each label value must match exactly as a string, or as a number within 1e-6; a question is correct only
if EVERY label answer is right; extra tags are ignored. (Stricter in one way: the official script skips questions with no
response, we count them wrong.)

usage: python research/ext_run.py <suite_dir> <lean|naive|naivemap|pi> <model> <out.jsonl> [limit] [ws_dir] [tasks_file]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from evaluate import CALL_CAP, CapHit, Meter, load_harness  # noqa: E402
from leanloop import lean  # noqa: E402
from run_pi import PI_CLI, parse as parse_pi  # noqa: E402

TAG = re.compile(r"@(\w+)\[(.*?)\]")


def dabench_score(final: str, labels: list) -> tuple[bool, dict]:
    got = dict(TAG.findall(final or ""))
    detail = {}
    for name, want in labels:
        v = got.get(name)
        ok = False
        if v is not None:
            if v == want:
                ok = True
            else:
                try:
                    ok = abs(float(v) - float(want)) < 1e-6
                except ValueError:
                    ok = False
        detail[name] = ok
    return bool(detail) and all(detail.values()), detail


def run_pi(prompt, work, model):
    env = dict(os.environ, PI_CODING_AGENT_DIR=str(ROOT / "pi-agent"))
    try:
        r = subprocess.run(["node", str(PI_CLI), "--provider", "ollama", "--model", model, "--no-session", "--thinking", "off",
                            "--mode", "json", prompt], cwd=work, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900)
        res = parse_pi(r.stdout)
        return {"final": res["final"], "calls": res["steps"], "prompt": res["prompt_tokens"], "completion": res["completion_tokens"],
                "tier": None, "error": None, "notes": [], "embed": 0, "trace": res["trace"]}
    except subprocess.TimeoutExpired:
        return {"final": "", "calls": 0, "prompt": 0, "completion": 0, "tier": None, "error": "timeout 900s", "notes": [], "embed": 0}


def run_ours(harness, prompt, work, model):
    meter, notes, tier, final, error, embed = Meter(model), [], None, "", None, 0
    events = []     # every progress event the harness emitted, in order (for trajectory export)

    def on_event(kind, text):
        events.append([kind, text if len(text) <= 1500 else text[:1500] + " ...[cut]"])
        if kind == "note":
            notes.append(text)
    def marked_chat(*a, **k):                    # mark where each model call starts, so a trajectory step == one model call
        events.append(["model_call", len(meter.log)])
        return meter(*a, **k)
    try:
        if harness == "lean":
            res = lean.run(prompt, work, model, chat=marked_chat, emit=on_event)
        else:
            res = load_harness(harness, model)(prompt, work, marked_chat)
        final, tier, embed = res.get("final", "") or "", res.get("tier"), res.get("embed_tokens", 0)
    except CapHit:
        error = f"call cap {CALL_CAP}"
    except Exception as e:  # a crashing harness scores the question as wrong
        error = f"{type(e).__name__}: {e}"
    if meter.overflow:
        error = error or "context overflow"
    return {"final": final, "calls": meter.calls, "prompt": meter.prompt, "completion": meter.completion, "tier": tier,
            "error": error, "notes": notes, "embed": embed, "events": events, "per_call": meter.log}


def main(suite, harness, model, out_path, limit=None, wsdir="ws", tasks_name="tasks.json"):
    suite = Path(suite)
    tasks = json.loads((suite / tasks_name).read_text(encoding="utf-8"))[: int(limit) if limit else None]
    rows = []
    if os.environ.get("EXT_RESUME") == "1" and Path(out_path).exists():      # continue an interrupted run: questions are independent
        rows = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        print(f"resuming: {len(rows)} questions already done", flush=True)
    done = {r["task"] for r in rows}
    t_all = time.time()
    for t in tasks:
        if t["id"] in done:
            continue
        with tempfile.TemporaryDirectory(prefix="ext_") as tmp:
            work = Path(tmp) / "ws"
            shutil.copytree(suite / wsdir / t["id"], work)
            t0 = time.time()
            r = run_pi(t["prompt"], work, model) if harness == "pi" else run_ours(harness, t["prompt"], work, model)
        if "labels" in t:                                   # DABench: scored here with the official rule
            ok, detail = dabench_score(r["final"], t["labels"])
            ok = ok and not r["error"]
        else:                                               # BIPIA: utility and attack success are scored afterwards by bipia_judge.py
            ok, detail = None, {}
        row = {"task": t["id"], "level": t.get("level", t.get("category")), "correct": ok, "calls": r["calls"], "prompt_tokens": r["prompt"],
               "completion_tokens": r["completion"], "tokens": r["prompt"] + r["completion"], "tier": r["tier"], "embed_tokens": r["embed"],
               "seconds": round(time.time() - t0, 1), "error": r["error"], "blocked": [n for n in r["notes"] if n.startswith("not run")],
               "final": (r["final"] or "")[:300], "detail": detail}
        for k in ("events", "per_call", "trace"):          # step-by-step record, if the harness produced one
            if k in r:
                row[k] = r[k]
        rows.append(row)
        print(f"  {t['id']:>4} {str(row['level'])[:12]:12} {'ok' if ok else '--' if ok is False else '  '} calls={row['calls']:2} tokens={row['tokens']:6} "
              f"tier={row['tier']} {row['seconds']:5.1f}s {row['error'] or ''}", flush=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
    n, ok = len(rows), sum(bool(r["correct"]) for r in rows)
    print("---")
    print(f"correct:  {ok}/{n}")
    print(f"tokens:   {sum(r['tokens'] for r in rows)}")
    print(f"calls:    {sum(r['calls'] for r in rows)}")
    print(f"seconds:  {round(time.time() - t_all, 1)}")
    for lv in ("easy", "medium", "hard"):
        g = [r for r in rows if r["level"] == lv]
        print(f"  {lv:6} {sum(r['correct'] for r in g)}/{len(g)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
