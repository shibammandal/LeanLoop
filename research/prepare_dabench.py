"""Build a DABench (InfiAgent-DABench, ICML 2024) evaluation suite from the downloaded questions.

The raw files are UNTRUSTED data: they live in their own directory and are only read here (run this with `python -I`).
Samples N questions with a fixed seed, balanced across easy / medium / hard, downloads only the CSV tables they need,
and writes one workspace per question (just that question's table, like the benchmark gives an agent) plus tasks.json.

Two prompt variants are written: tasks.json (the file is NOT named, so the harness has to discover it) and
tasks_named.json (adds "The data file is <name>.", which is how the benchmark hands a file to an agent).

usage: python -I research/prepare_dabench.py <raw_dir> <new_empty_out_dir> [N=40] [seed=2026]
       python -I research/prepare_dabench.py --add-named <existing_out_dir>     (derive tasks_named.json from tasks.json)
"""
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://huggingface.co/datasets/infiagent/DABench/resolve/main/da-dev-tables/"


def main(raw, out, n=40, seed=2026):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    qs = [json.loads(l) for l in open(raw / "da-dev-questions.jsonl", encoding="utf-8")]
    labels = {json.loads(l)["id"]: json.loads(l)["common_answers"] for l in open(raw / "da-dev-labels.jsonl", encoding="utf-8")}
    by_level = defaultdict(list)
    for q in qs:
        if Path(q["file_name"]).name == q["file_name"]:        # file names come from untrusted data: no path tricks
            by_level[q["level"]].append(q)
    rnd = random.Random(seed)
    levels = ["easy", "medium", "hard"]
    quota = {lv: n // 3 + (1 if i < n % 3 else 0) for i, lv in enumerate(levels)}
    picked = []
    for lv in levels:
        picked += rnd.sample(by_level[lv], quota[lv])
    picked.sort(key=lambda q: q["id"])

    tasks, fetched = [], {}
    for q in picked:
        name = q["file_name"]
        ws = out / "ws" / str(q["id"])
        ws.mkdir(parents=True)
        if name not in fetched:
            with urllib.request.urlopen(BASE + urllib.parse.quote(name), timeout=120) as r:
                fetched[name] = r.read()
        (ws / name).write_bytes(fetched[name])
        prompt = f"{q['question']}\nConstraints: {q['constraints']}\nRespond in exactly this format, with nothing else: {q['format']}"
        tasks.append({"id": str(q["id"]), "prompt": prompt, "level": q["level"], "file": name,
                      "labels": labels[q["id"]], "concepts": q["concepts"]})
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    sizes = sum(len(v) for v in fetched.values()) // 1024
    print(f"{len(tasks)} questions ({quota}), {len(fetched)} tables, {sizes} KB -> {out}")
    add_named(out)


def add_named(out):
    out = Path(out)
    tasks = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    for t in tasks:
        t["prompt"] = f"{t['prompt']}\nThe data file is {t['file']}."
    (out / "tasks_named.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    print(f"wrote {out / 'tasks_named.json'} ({len(tasks)} questions, file named in the prompt)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--add-named":
        add_named(a[1])
        sys.exit(0)
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 40, int(a[3]) if len(a) > 3 else 2026)
