"""Summarize results/<prefix><harness>.jsonl: accuracy, tokens, model calls, time, and an illustrative dollar cost.

usage: python compare.py [prefix]      e.g.  python compare.py test_   (held-out test split)
Cost uses a reference API price (illustrative, not a quote): $3 / 1M input tokens, $15 / 1M output tokens.
"""
import json
import sys
from pathlib import Path

IN_PRICE, OUT_PRICE = 3 / 1e6, 15 / 1e6
prefix = sys.argv[1] if len(sys.argv) > 1 else ""
d = Path(__file__).parent / "results"
names = ("pi", "naive", "lean0", "lean")


def norm(r):  # pi rows and evaluate.py rows use slightly different field names
    return {"task": r["task"], "correct": bool(r["correct"]), "calls": r.get("calls", r.get("steps", 0)),
            "prompt": r["prompt_tokens"], "completion": r["completion_tokens"],
            "total": r.get("tokens", r.get("total_tokens", 0)), "seconds": r.get("seconds", 0)}


runs = {m: [norm(json.loads(l)) for l in open(d / f"{prefix}{m}.jsonl", encoding="utf-8")]
        for m in names if (d / f"{prefix}{m}.jsonl").exists()}

print(f"{'harness':8}{'correct':>9}{'calls':>7}{'prompt':>9}{'compl':>8}{'total':>9}{'tok/task':>10}{'seconds':>9}{'$ (ref)':>9}")
tot = {}
for m, rs in runs.items():
    s = {k: sum(r[k] for r in rs) for k in ("calls", "prompt", "completion", "total", "seconds")}
    s["ok"], s["n"] = sum(r["correct"] for r in rs), len(rs)
    s["cost"] = s["prompt"] * IN_PRICE + s["completion"] * OUT_PRICE
    tot[m] = s
    print(f"{m:8}{s['ok']:>6}/{s['n']:<2}{s['calls']:>7}{s['prompt']:>9}{s['completion']:>8}{s['total']:>9}"
          f"{s['total'] // s['n']:>10}{s['seconds']:>9.0f}{s['cost']:>9.4f}")

if "lean" in tot:
    for base in ("pi", "naive", "lean0"):
        if base in tot:
            L, B = tot["lean"], tot[base]
            print(f"lean vs {base:6}: tokens {1 - L['total'] / B['total']:+.0%} saved, cost {1 - L['cost'] / B['cost']:+.0%} saved, "
                  f"calls {L['calls'] - B['calls']:+d}, time {L['seconds'] - B['seconds']:+.0f}s, correct {L['ok']} vs {B['ok']}")

print("\nper task (total tokens, ok/--):")
ids = [r["task"] for r in next(iter(runs.values()))]
print(f"{'task':22}" + "".join(f"{m:>14}" for m in runs))
for i, tid in enumerate(ids):
    print(f"{tid:22}" + "".join(f"{runs[m][i]['total']:>10}{'  ok' if runs[m][i]['correct'] else '  --'}" for m in runs))
