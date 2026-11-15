"""Summarize the external-benchmark runs (DABench) straight from the saved result files.

Two prompt variants of the same 40 questions:
  named    "The data file is <name>."   (the benchmark's own protocol; the headline numbers)
  unnamed  no file name                  (the harness has to discover the file; an ablation)
For each pair of harnesses on the same model it also prints a PAIRED comparison on the same questions, with an exact
two-sided sign-test p-value, because at 40 questions a gap of a few answers is often noise. Scores are the official
closed-form rule (strict); a lenient score (right values anywhere in the reply) is shown to rule out format artifacts.
usage: python demo/ext_summary.py
"""
import json
import re
from math import comb
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
_TASKS_FILE = R.parent / "data" / "dabench" / "tasks.json"     # the benchmark's labels; not shipped (download them: see SUBMISSION.md)
TASKS = {t["id"]: t for t in json.loads(_TASKS_FILE.read_text(encoding="utf-8"))} if _TASKS_FILE.exists() else {}
MODELS = [("Qwen3 8B", "qwen3"), ("Qwen2.5-Coder 7B", "coder")]
HARNESSES = ["lean", "naive", "pi", "pi2"]
PRETTY = {"pi2": "pi (run 2, traced)"}
def _named(tag, h):
    if h == "pi2":      # the traced second pi run exists for Qwen3 only (pi cannot run Qwen2.5-Coder)
        return R / "traj_dabench_qwen3_pi.jsonl" if tag == "qwen3" else R / "no_such_run.jsonl"
    return R / f"dabench_named_{tag}_{h}.jsonl"


VARIANTS = [("file named in the prompt (benchmark protocol)", "named", _named),
            ("file NOT named (harness must find it)", "unnamed", lambda tag, h: R / "dabench_v1_unnamed" / f"{tag}_{h}.jsonl")]


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else None


def sign_test_p(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def lenient(final, labels):
    if not labels:
        return None
    nums = set(re.findall(r"-?\d+(?:\.\d+)?", final or ""))
    low = (final or "").lower()
    for _, want in labels:
        w = want.strip()
        if not w:
            continue
        try:
            if not any(abs(float(x) - float(w)) < 1e-6 for x in nums):
                return False
        except ValueError:
            if w.lower() not in low:
                return False
    return True


def main():
    for title, key, path in VARIANTS:
        print(f"\n## DABench, 40 questions (seed 2026, balanced easy/medium/hard): {title}\n")
        print("| model | harness | correct | lenient | easy | medium | hard | calls | calls/q | total tokens | tokens/q | tokens per correct | answered by tier 1 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        data = {}
        for label, tag in MODELS:
            for h in HARNESSES:
                rs = rows(path(tag, h))
                if not rs:
                    continue
                data[(tag, h)] = {r["task"]: r for r in rs}
                n, ok = len(rs), sum(r["correct"] for r in rs)
                lv = {k: f"{sum(r['correct'] for r in rs if r['level'] == k)}/{sum(r['level'] == k for r in rs)}" for k in ("easy", "medium", "hard")}
                tok, calls = sum(r["tokens"] for r in rs), sum(r["calls"] for r in rs)
                t1 = f"{sum(r['tier'] == 1 for r in rs)}/{n}" if h == "lean" else "-"
                print(f"| {label} | {PRETTY.get(h, h)} | {ok}/{n} | {(str(sum(lenient(r['final'], TASKS[r['task']]['labels']) for r in rs)) + '/' + str(n)) if TASKS else 'n/a (labels not downloaded)'} | {lv['easy']} | "
                      f"{lv['medium']} | {lv['hard']} | {calls} | {calls / n:.1f} | {tok:,} | {tok // n:,} | {tok // max(ok, 1):,} | {t1} |")
        print("\nPaired comparison on the same questions (exact sign test; small n):\n")
        for label, tag in MODELS:
            a = data.get((tag, "lean"))
            for h in HARNESSES[1:]:
                b = data.get((tag, h))
                if not a or not b:
                    continue
                ids = [i for i in a if i in b]
                lo = sum(a[i]["correct"] and not b[i]["correct"] for i in ids)
                bo = sum(b[i]["correct"] and not a[i]["correct"] for i in ids)
                ta, tb = sum(a[i]["tokens"] for i in ids), sum(b[i]["tokens"] for i in ids)
                ca, cb = sum(a[i]["calls"] for i in ids), sum(b[i]["calls"] for i in ids)
                print(f"- {label}: leanloop vs {PRETTY.get(h, h)}: leanloop-only correct {lo}, {PRETTY.get(h, h)}-only correct {bo} (p={sign_test_p(lo, bo):.2f}); "
                      f"tokens {1 - ta / tb:+.0%} ({ta:,} vs {tb:,}); calls {ca} vs {cb}")
        for label, tag in MODELS:
            a = data.get((tag, "lean"))
            if a:
                esc = [r for r in a.values() if r["tier"] != 1]
                blocked = sum(bool(r.get("blocked")) for r in a.values())
                print(f"- {label} leanloop: {len(esc)}/{len(a)} questions needed the fallback loop and used "
                      f"{sum(r['tokens'] for r in esc):,} of its {sum(r['tokens'] for r in a.values()):,} tokens; "
                      f"{blocked} had a program blocked by the read-only check")


def fisher_p(a, b, n=40):
    """Exact two-sided Fisher p-value for a/n versus b/n."""
    total = a + b
    denom = comb(2 * n, total)
    probs = {k: comb(n, k) * comb(n, total - k) / denom for k in range(max(0, total - n), min(n, total) + 1)}
    return sum(p for p in probs.values() if p <= probs[a] + 1e-12)


def bipia():
    print("\n## BIPIA indirect prompt injection (Microsoft): 40 documents (20 email, 20 table), one planted instruction each\n")
    runs = [("leanloop", "Qwen3 8B", "bipia_qwen3_lean_scored.json"), ("leanloop", "Qwen2.5-Coder 7B", "bipia_coder_lean_scored.json"),
            ("pi (unmodified)", "Qwen3 8B", "bipia_qwen3_pi_scored.json")]
    print("| harness | model | planted instruction followed | still answered the question | same questions, no attack | tokens | calls | judge catches real completions |")
    print("|---|---|---|---|---|---|---|---|")
    S = {}
    for h, m, f in runs:
        p = R / f
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        S[(h, m)] = s
        j = s["judge"]
        print(f"| {h} | {m} | {s['asr']}/{s['n']} | {s['utility']}/{s['n']} | {s['clean_utility']}/{s['n']} | "
              f"{sum(r['tokens'] for r in s['rows']):,} | {sum(r['calls'] for r in s['rows'])} | {j['sensitivity_on_task_done']}/{j['n']} |")
    ab = R / "bipia_ablation_fullctx_qwen3.json"
    if ab.exists():
        a = json.loads(ab.read_text(encoding="utf-8"))
        print(f"| whole document + leanloop's strict prompt (ablation) | Qwen3 8B | {a['asr']}/{a['n']} | {a['utility']}/{a['n']} | - | {a['tokens']:,} | {a['calls']} | - |")
    pi, ln = S.get(("pi (unmodified)", "Qwen3 8B")), S.get(("leanloop", "Qwen3 8B"))
    if pi and ln:
        print(f"\n- followed the planted instruction: pi {pi['asr']}/40 vs leanloop {ln['asr']}/40 (Fisher exact p={fisher_p(pi['asr'], ln['asr']):.5f})")
        if ab.exists():
            print(f"- pi {pi['asr']}/40 vs whole document + strict prompt {a['asr']}/40 (p={fisher_p(pi['asr'], a['asr']):.3f}); "
                  f"whole document {a['asr']}/40 vs leanloop's pruned {ln['asr']}/40 (p={fisher_p(a['asr'], ln['asr']):.2f}): pruning's extra effect is not significant")
        tok_p, tok_l = sum(r["tokens"] for r in pi["rows"]), sum(r["tokens"] for r in ln["rows"])
        print(f"- tokens: leanloop {tok_l:,} vs pi {tok_p:,} ({1 - tok_l / tok_p:.0%} fewer)")
        bipia_tasks = R.parent / "data" / "bipia" / "tasks.json"
        for src in ("email", "table") if bipia_tasks.exists() else ():
            tk = {t["id"]: t["source"] for t in json.loads(bipia_tasks.read_text(encoding="utf-8"))}
            f = lambda s, key: sum(bool(r[key]) for r in s["rows"] if tk[r["task"]] == src)    # noqa: E731
            print(f"- clean-copy accuracy on {src} questions: pi {f(pi, 'clean_utility')}/20 vs leanloop {f(ln, 'clean_utility')}/20")
        if not bipia_tasks.exists():
            print("- (per-source accuracy needs data/bipia/tasks.json, which is rebuilt by research/prepare_bipia.py)")
        sens = ln["judge"]["sensitivity_on_task_done"] / ln["judge"]["n"]
        print("- what true attack rate is still plausible after seeing 0/40 from a judge that catches "
              f"{sens:.0%} of real completions: " + "; ".join(f"{r:.0%} -> P(0/40)={(1 - r * sens) ** 40:.0%}" for r in (0.05, 0.10, 0.15)))
    rp = R / "bipia_reach.json"
    if rp.exists():
        r = json.loads(rp.read_text(encoding="utf-8"))
        print("- did the planted text reach leanloop's model? " + "; ".join(
            f"{k}: {v['reached_the_model']}/{v['n']} (removed by guard {v['removed_by_guard']})" for k, v in r.items()))


def agentdojo():
    print("\n## AgentDojo feasibility probe (ETH Zurich): banking suite, Qwen3 8B through AgentDojo's own tool-calling pipeline\n")
    print("This is the model + AgentDojo's pipeline, NOT leanloop (no adapter exists for its tool environments).\n")
    for name, f in (("no attack", "agentdojo_banking_utility.log"), ("important_instructions attack", "agentdojo_banking_attack.log")):
        p = R / f
        if p.exists():
            print(f"- {name}: " + " | ".join(l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()))


def auditkit():
    p = R / "auditkit_dabench.json"
    if not p.exists():
        return
    res = json.loads(p.read_text(encoding="utf-8"))
    print("\n## Lexsi AuditKIT cross-check: DABench trajectories exported in AgentTune's format, scored by AuditKIT's agent evaluation\n")
    print("Verdicts come from AuditKIT's runner (the DABench rule as a versioned `CustomPredicate`); tokens come from our recorded counts, "
          "placed in AuditKIT's own `resource_use.tokens` slot. A step is one model call.\n")
    budgets = ["1", "2", "3", "4", "6", "10"]
    print("| harness | AuditKIT success | agrees with our scorer | " + " | ".join(f"within {b} call{'s' if b != '1' else ''}" for b in budgets) + " | tokens | tokens per success |")
    print("|---|---|---|" + "---|" * len(budgets) + "---|---|")
    for r in res:
        print(f"| {r['label']} | {r['auditkit_success']}/{r['n']} | {r['agree_with_own_scorer']}/{r['n']} | "
              + " | ".join(str(r['budgets'][b]) for b in budgets) + f" | {r['tokens_total']:,} | {r['tokens_per_success']:,} |")


def table_routing():
    runs = [("leanloop, table routing OFF (before)", "bipia_fresh_lean_old_scored.json"), ("leanloop, table routing ON (after)", "bipia_fresh_lean_new_scored.json"),
            ("pi (unmodified)", "bipia_fresh_pi_scored.json")]
    if not all((R / f).exists() for _, f in runs):
        return
    S = {name: json.loads((R / f).read_text(encoding="utf-8")) for name, f in runs}
    print("\n## Table-routing fix, validated on 40 FRESH BIPIA documents (none used before; Qwen3 8B)\n")
    print("| | planted instruction followed | still answered (attacked) | same documents, no attack | tokens | calls |")
    print("|---|---|---|---|---|---|")
    for name, s in S.items():
        print(f"| {name} | {s['asr']}/{s['n']} | {s['utility']}/{s['n']} | {s['clean_utility']}/{s['n']} | "
              f"{sum(r['tokens'] for r in s['rows']):,} | {sum(r['calls'] for r in s['rows'])} |")
    tasks_file = R.parent / "data" / "bipia_fresh" / "tasks.json"
    if tasks_file.exists():
        src = {t["id"]: t["source"] for t in json.loads(tasks_file.read_text(encoding="utf-8"))}
        for kind in ("table", "email"):
            cells = {name: sum(bool(r["clean_utility"]) for r in s["rows"] if src[r["task"]] == kind) for name, s in S.items()}
            print(f"- clean-copy accuracy on {kind} questions (of 20): " + "; ".join(f"{n.split(',')[0] if 'leanloop' in n else n}"
                  f"{' ' + n.split(', ')[1] if 'leanloop' in n else ''}: {v}" for n, v in cells.items()))
        old, new = S[runs[0][0]], S[runs[1][0]]
        a = sum(bool(r["clean_utility"]) for r in new["rows"] if src[r["task"]] == "table")
        b = sum(bool(r["clean_utility"]) for r in old["rows"] if src[r["task"]] == "table")
        print(f"- table accuracy after vs before: {a}/20 vs {b}/20 (Fisher exact p={fisher_p(a, b, 20):.3f})")
    print(f"- planted instruction followed: after {S[runs[1][0]]['asr']}/40, before {S[runs[0][0]]['asr']}/40, pi {S[runs[2][0]]['asr']}/40")
    first_pi, first_ln = R / "bipia_qwen3_pi_scored.json", R / "bipia_qwen3_lean_scored.json"
    pi_f, ln_f = S[runs[2][0]], S[runs[0][0]]
    print(f"- fresh set, pi {pi_f['asr']}/40 vs leanloop (as validated) {ln_f['asr']}/40: Fisher exact p={fisher_p(pi_f['asr'], ln_f['asr']):.4f}")
    if first_pi.exists() and first_ln.exists():
        p1, l1 = json.loads(first_pi.read_text(encoding="utf-8")), json.loads(first_ln.read_text(encoding="utf-8"))
        a, b = p1["asr"] + pi_f["asr"], l1["asr"] + ln_f["asr"]
        ca, cb = p1["clean_utility"] + pi_f["clean_utility"], l1["clean_utility"] + ln_f["clean_utility"]
        print(f"- both sets pooled (80 documents): planted instruction followed pi {a}/80 ({a / 80:.0%}) vs leanloop {b}/80 ({b / 80:.0%}), "
              f"p={fisher_p(a, b, 80):.1e}")
        print(f"- both sets pooled: accuracy on the clean copies pi {ca}/80 ({ca / 80:.0%}) vs leanloop {cb}/80 ({cb / 80:.0%}), p={fisher_p(ca, cb, 80):.3f}")
        tp, tl = sum(r["tokens"] for r in pi_f["rows"]), sum(r["tokens"] for r in ln_f["rows"])
        print(f"- tokens: leanloop uses {1 - tl / tp:.0%} fewer than pi on the fresh set")


def safetune():
    p = R / "safetune_guardrail.json"
    if not p.exists():
        return
    r = json.loads(p.read_text(encoding="utf-8"))
    n = r["n_test_each"]
    print(f"\n## Lexsi SafeTune: can refusal-direction steering restore refusals that an attack removed? "
          f"(separate experiment; {r['model']} on CPU; {n} harmful + {n} benign test prompts; attack = {r['attack']})\n")
    print("Validation sweep (steering settings tried; the best by 'refuses attacked-harmful minus refuses benign' was chosen, never using the test prompts):\n")
    print("| setting | refuses attacked-harmful | refuses benign (over-refusal) | score |")
    print("|---|---|---|---|")
    for k, v in r["validation_sweep"].items():
        print(f"| {k}{' **(chosen)**' if k == r['chosen_setting'] else ''} | {v['refuse_harmful_under_attack']:.0%} | {v['refuse_benign']:.0%} | {v['score']:+.2f} |")
    t = r["test"]
    base = t["A_attack_no_safeguard"]["mean_prompt_tokens"]            # the system prompt's cost = MEASURED mean prompt length minus the attacked baseline's
    names = {"A0_no_attack_no_safeguard": "no attack, no safeguard (the model's normal behaviour)", "A_attack_no_safeguard": "attack, no safeguard",
             "B_attack_safety_prompt": "attack + safety system prompt", "C_attack_steering": f"attack + SafeTune steering ({r['chosen_setting']})"}
    print("\nTest prompts:\n")
    print("| condition | refuses harmful requests | refuses benign requests (over-refusal) | extra prompt tokens per request |")
    print("|---|---|---|---|")
    for k, v in t.items():
        extra = 0 if k.startswith("A") else round(v["mean_prompt_tokens"] - base, 1)
        print(f"| {names[k]} | {v['refuse_harmful']:.0%} | {v['refuse_benign']:.0%} | {extra} |")
    print(f"\nEmpty outputs: {r['empty_outputs']}. Runtime {r['seconds'] // 60} minutes.")


if __name__ == "__main__":
    main()
    bipia()
    table_routing()
    auditkit()
    safetune()
    agentdojo()
