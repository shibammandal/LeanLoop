"""Re-score saved result files with the current check() (same rule for every harness).

usage: python research/rescore.py <tasks.json> <results.jsonl> [more.jsonl ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import check  # noqa: E402

tasks = {t["id"]: t["expect"] for t in json.loads(Path(sys.argv[1]).read_text())}
for path in sys.argv[2:]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    before = sum(r["correct"] for r in rows)
    for r in rows:
        r["correct"] = check(r.get("final", ""), tasks[r["task"]]) and not r.get("error") and not r.get("overflow")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"{path}: {before} -> {sum(r['correct'] for r in rows)} correct")
