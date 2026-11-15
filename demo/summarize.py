"""Single source of truth for every number in SUBMISSION.md, README.md and the charts.

Reads the saved result files in results/ and prints markdown tables. Nothing here is typed by hand.
usage: python demo/summarize.py            (prints tables)   |   import summarize; summarize.numbers()
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
IN_PRICE, OUT_PRICE = 3 / 1e6, 15 / 1e6          # illustrative reference API price, not a quote

# the five held-out runs of leanloop made while the code was being built (the "hint" run was a reverted experiment)
LEAN_RUNS = ["test_lean.jsonl", "test_lean_after_docs.jsonl", "test_lean_robust.jsonl", "test_lean_robust3.jsonl",
             "test_lean_final.jsonl"]


def rows(name):
    return [json.loads(l) for l in open(R / name, encoding="utf-8")]


def tot(rs):
    g = lambda r, *ks: next((r[k] for k in ks if k in r), 0)          # pi rows and evaluate rows name fields differently
    return {"n": len(rs), "ok": sum(bool(r["correct"]) for r in rs), "calls": sum(g(r, "calls", "steps") for r in rs),
            "tokens": sum(g(r, "tokens", "total_tokens") for r in rs), "prompt": sum(r["prompt_tokens"] for r in rs),
            "completion": sum(r["completion_tokens"] for r in rs), "seconds": sum(r.get("seconds", 0) for r in rs)}


def cost(t):
    return t["prompt"] * IN_PRICE + t["completion"] * OUT_PRICE


def numbers():
    base = {"pi": tot(rows("test_pi.jsonl")), "naive": tot(rows("test_naive.jsonl")), "lean0": tot(rows("test_lean0.jsonl"))}
    runs = [tot(rows(f)) for f in LEAN_RUNS]
    worst, best = max(runs, key=lambda t: t["tokens"]), min(runs, key=lambda t: t["tokens"])
    docs = {}
    for f in ("docs_bench.jsonl", "docs_bench_robust.jsonl"):
        for r in rows(f):
            d = docs.setdefault(r["mode"], {"runs": {}})["runs"].setdefault(f, {"n": 0, "ok": 0, "tok": 0, "inj": 0, "poi": 0})
            d["n"] += 1
            d["ok"] += r["correct"]
            d["tok"] += r["llm_tokens"]
            d["inj"] += r["injected"]
            d["poi"] += r["poisoned"]
    return {"base": base, "lean_runs": runs, "lean_worst": worst, "lean_best": best, "docs": docs}


def main():
    n = numbers()
    b, runs, w, bst = n["base"], n["lean_runs"], n["lean_worst"], n["lean_best"]
    print("## Held-out test (12 tasks): tokens, correctness, model calls\n")
    print("| harness | correct | model calls | calls/task | total tokens | tokens/task | tokens per correct answer | $ ref |")
    print("|---|---|---|---|---|---|---|---|")
    for k, label in (("pi", "pi (unmodified)"), ("naive", "naive loop"), ("lean0", "leanloop v0 (paging only)")):
        t = b[k]
        print(f"| {label} | {t['ok']}/{t['n']} | {t['calls']} | {t['calls'] / t['n']:.1f} | {t['tokens']:,} | "
              f"{t['tokens'] // t['n']:,} | {t['tokens'] // max(t['ok'], 1):,} | ${cost(t):.3f} |")
    lo, hi = bst, w
    print(f"| **leanloop** (5 runs: best / worst) | {lo['ok']}/{lo['n']} (all 5 runs) | {lo['calls']}–{hi['calls']} | "
          f"{lo['calls'] / lo['n']:.1f}–{hi['calls'] / hi['n']:.1f} | {lo['tokens']:,}–{hi['tokens']:,} | "
          f"{lo['tokens'] // lo['n']:,}–{hi['tokens'] // hi['n']:,} | {lo['tokens'] // lo['ok']:,}–{hi['tokens'] // hi['ok']:,} | "
          f"${cost(lo):.3f}–{cost(hi):.3f} |")
    print(f"\nall leanloop runs: correct={[t['ok'] for t in runs]} tokens={[t['tokens'] for t in runs]} calls={[t['calls'] for t in runs]}")
    pi = b["pi"]
    print(f"\nvs pi: worst run saves {1 - hi['tokens'] / pi['tokens']:.1%} of tokens, best {1 - lo['tokens'] / pi['tokens']:.1%}; "
          f"calls {hi['calls']}–{lo['calls']} vs {pi['calls']}")
    print(f"tokens per correct answer: pi {pi['tokens'] // pi['ok']:,} vs leanloop {lo['tokens'] // lo['ok']:,}–{hi['tokens'] // hi['ok']:,} "
          f"({pi['tokens'] / pi['ok'] / (hi['tokens'] / hi['ok']):.0f}x–{pi['tokens'] / pi['ok'] / (lo['tokens'] / lo['ok']):.0f}x fewer)")
    nv = b["naive"]
    print(f"vs naive loop (same file Q&A tools, full history): worst run saves {1 - hi['tokens'] / nv['tokens']:.0%}, "
          f"best {1 - lo['tokens'] / nv['tokens']:.0%}")
    print("\n## Documents + prompt injection (24 runs per mode per bench run; two bench runs)\n")
    print("| context sent | run | correct | generic injection won | topical poisoning won | model tokens/question |")
    print("|---|---|---|---|---|---|")
    for mode, label in (("full", "all documents"), ("rag2", "top-2 whole docs"), ("span", "micro-spans"), ("span_guard", "micro-spans + guard")):
        for i, (f, d) in enumerate(n["docs"][mode]["runs"].items(), 1):
            print(f"| {label} | {i} | {d['ok']}/{d['n']} | {d['inj']}/{d['n']} | {d['poi']}/{d['n']} | {d['tok'] // d['n']:,} |")


if __name__ == "__main__":
    main()
