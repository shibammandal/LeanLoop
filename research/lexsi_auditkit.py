"""Score exported trajectories with Lexsi Labs' AuditKIT (agent_eval), as an independent cross-check, and add what it lacks.

What this does, using only AuditKIT's public API:
  1. reads each exported run folder with `episodes_from_agenttune` (AuditKIT's AgentTune reader);
  2. scores every episode with the DABench closed-form rule, expressed as an AuditKIT outcome (`CustomPredicate`, versioned),
     so the verdicts come from AuditKIT's runner and not from our own scorer;
  3. fills AuditKIT's `resource_use.tokens` slot (always empty for AgentTune files, which log no tokens) from our recorded counts;
  4. uses AuditKIT's per-case BUDGETS (`max_turns`) to ask Track 3's question directly: how many questions are still solved
     within N model calls?  (A step == one model call, for every harness.)
Run it with a Python that has `auditkit` installed (pip install auditkit), e.g. a separate virtual environment.

usage: python -I research/lexsi_auditkit.py <tasks.json> <out.json> <label>=<run_folder> [<label>=<run_folder> ...]
       python -I research/lexsi_auditkit.py --selfcheck          (tiny synthetic round trip; no data needed)
"""
import json
import re
import sys
import tempfile
from pathlib import Path

from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, CustomPredicate, episodes_from_agenttune

TAG = re.compile(r"@(\w+)\[(.*?)\]")
BUDGETS = [1, 2, 3, 4, 6, 10]          # max model calls


def dabench_ok(final, labels):
    """DABench's official closed-form rule: every label tag present, exact string or numeric within 1e-6."""
    got = dict(TAG.findall(final or ""))
    for name, want in labels:
        v = got.get(name)
        if v is None:
            return False
        if v != want:
            try:
                if not abs(float(v) - float(want)) < 1e-6:
                    return False
            except ValueError:
                return False
    return bool(labels)


def predicate(episode, case):
    return dabench_ok(episode.final_answer, case.metadata["labels"])


def score(label, folder, tasks):
    folder = Path(folder)
    raw = {}
    for line in open(folder / "trajectories.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        raw[rec["trajectory_id"]] = rec
    by_id = {t["id"]: t for t in tasks}
    out = {"label": label, "n": 0, "budgets": {}}
    for budget in [None] + BUDGETS:
        episodes = episodes_from_agenttune(str(folder))          # fresh episodes for each run
        cases = []
        for ep in episodes:
            qid = ep.source_id.split("-", 1)[1]
            rec = raw[ep.source_id]
            ep.case_id = qid
            use = rec["metadata"]["usage"]
            ep.counters["tokens"] = use["total_tokens"]                    # the slot AuditKIT has but AgentTune files never fill
            ep.counters["latency_ms"] = (rec["metadata"].get("seconds") or 0) * 1000
            t = by_id[qid]
            cases.append(AgentCase(id=qid, task=t["prompt"], category=t.get("level", "generic"),
                                   outcome=CustomPredicate(predicate, version="dabench-closed-form-1"),
                                   budgets={"max_turns": budget} if budget else {}, metadata={"labels": t["labels"]}))
        result = AgentEvalRunner().run(AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes, scorers=[], trials=1))
        rows = result.rows
        solved = [r for r in rows if r.outcome["verdict"] == "success" and r.status == "completed"]
        key = "no_budget" if budget is None else str(budget)
        out["budgets"][key] = len(solved)
        if budget is None:
            out["n"] = len(rows)
            out["auditkit_success"] = sum(r.outcome["verdict"] == "success" for r in rows)
            out["agree_with_own_scorer"] = sum((r.outcome["verdict"] == "success") == bool(raw[next(k for k in raw if k.endswith("-" + r.case_id))]["metadata"].get("own_verdict"))
                                               for r in rows)
            toks = [r.resource_use.get("tokens") or 0 for r in rows]
            out["tokens_total"] = sum(toks)
            out["tokens_per_success"] = round(sum(toks) / max(1, out["auditkit_success"]))
            out["coverage"] = sorted({r.coverage_label for r in rows})
            out["n_tool_calls"] = sum(r.resource_use.get("n_tool_calls") or 0 for r in rows)
            out["statuses"] = {s: sum(r.status == s for r in rows) for s in sorted({r.status for r in rows})}
    return out


def selfcheck():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import lexsi_export
    tasks = [{"id": "1", "prompt": "q1", "level": "easy", "labels": [["a", "5"]]}]
    rows = [{"task": "1", "final": "@a[5]", "level": "easy", "tier": 1, "seconds": 2.0, "calls": 2, "prompt_tokens": 100,
             "completion_tokens": 20, "tokens": 120, "correct": True,
             "events": [["tier", "1: program-first"], ["model_call", 0], ["program", "print(1)"], ["output", "wrong"],
                        ["model_call", 1], ["program", "print('@a[5]')"], ["output", "@a[5]"], ["final", "@a[5]"]],
             "per_call": [{"prompt": 60, "completion": 10}, {"prompt": 40, "completion": 10}]}]
    with tempfile.TemporaryDirectory() as d:
        lexsi_export.export(rows, tasks, "leanloop", "m", d)
        eps = episodes_from_agenttune(d)
        assert len(eps) == 1 and eps[0].final_answer == "@a[5]", eps
        assert eps[0].counters["n_turns"] == 2 and eps[0].counters["n_tool_calls"] == 2, eps[0].counters
        res = score("selfcheck", d, tasks)
        assert res["auditkit_success"] == 1 and res["budgets"]["2"] == 1 and res["budgets"]["1"] == 0, res
        assert res["tokens_total"] == 120, res
    print("selfcheck ok: exported run read back by AuditKIT (2 turns, 2 tool calls), solved within 2 calls but not 1, tokens=120")


def main():
    if sys.argv[1:] == ["--selfcheck"]:
        return selfcheck()
    tasks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    results = [score(*arg.split("=", 1), tasks) for arg in sys.argv[3:]]
    Path(sys.argv[2]).write_text(json.dumps(results, indent=1), encoding="utf-8")
    for r in results:
        print(f"{r['label']}: AuditKIT success {r['auditkit_success']}/{r['n']} (agrees with our scorer on {r['agree_with_own_scorer']}/{r['n']}); "
              f"solved within N model calls: " + ", ".join(f"{k}:{v}" for k, v in r["budgets"].items() if k != "no_budget") +
              f"; tokens {r['tokens_total']:,} ({r['tokens_per_success']:,} per success)")


if __name__ == "__main__":
    main()
