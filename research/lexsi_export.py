"""Export recorded runs as AgentTune-format trajectories, so Lexsi's AuditKIT can score them.

AgentTune's trajectory record (what AuditKIT's `episodes_from_agenttune` reads) is one JSON object per line:
    {"trajectory_id", "task", "steps": [{"step_number", "thought", "action", "observation"}], "final_response",
     "reward": null, "metadata": {"conversation", "tool_call_count", ...}}
with `action` either a flat {"id","name","arguments"} call or {"tool_calls": [...]}. A run folder also carries
`lexsi_provenance.json`. This file is written by leanloop, NOT by AgentTune: it follows the documented/importable shape and is
checked by reading it back through AuditKIT (see research/lexsi_auditkit.py).

One step == one MODEL CALL for every harness, so "number of turns" means the same thing for leanloop and pi:
  leanloop: steps are rebuilt from the ordered progress events, split at the `model_call` markers ext_run.py inserts;
  pi:       steps are pi's own assistant messages (with their tool calls and tool results).
Token counts are not part of AgentTune's format (it logs none); they go in `metadata.usage`.

usage: python -I research/lexsi_export.py <rows.jsonl> <tasks.json> <leanloop|pi> <model> <out_dir>
"""
import json
import sys
from pathlib import Path


def _loads(arguments):
    if isinstance(arguments, dict):
        return arguments
    try:
        v = json.loads(arguments)
        return v if isinstance(v, dict) else {"value": v}
    except (TypeError, ValueError):
        return {"raw": arguments}


def _action(calls):
    """Flat call for one, {"tool_calls": [...]} for several, None for none (the step is a plain answer)."""
    if not calls:
        return None
    return calls[0] if len(calls) == 1 else {"tool_calls": calls}


def steps_from_events(events, per_call, prompt):
    """leanloop progress events -> one step per model call."""
    groups, pre = [], []
    for kind, text in events:
        if kind == "model_call":
            groups.append({"calls": [], "obs": [], "usage": per_call[len(groups)] if len(groups) < len(per_call) else None})
        elif kind in ("tier", "final"):
            continue
        elif not groups:
            pre.append((kind, text))               # e.g. retrieval happens before the model call that answers from it
        else:
            groups[-1].setdefault("later", []).append((kind, text))
    if pre:                                         # attach pre-call context to the first model call (or make a step for it)
        if not groups:
            groups.append({"calls": [], "obs": [], "usage": None})
        groups[0]["later"] = pre + groups[0].get("later", [])
    steps = []
    for i, g in enumerate(groups):
        calls, obs, n = [], [], 0
        for kind, text in g.get("later", []):
            if kind == "program":
                calls.append({"id": f"call_{i}_{n}", "name": "python", "arguments": {"code": text}})
            elif kind == "call":
                name, _, args = text.partition(" ")
                calls.append({"id": f"call_{i}_{n}", "name": name, "arguments": _loads(args)})
            elif kind == "excerpts":
                calls.append({"id": f"call_{i}_{n}", "name": "retrieve_spans", "arguments": {"query": prompt}})
                obs.append(text)
                n += 1
                continue
            elif kind == "output":
                obs.append(text)
                continue
            elif kind == "note":
                obs.append(f"[note] {text}")
                continue
            else:
                continue
            n += 1
        steps.append({"step_number": i, "thought": None, "action": _action(calls),
                      "observation": "\n".join(obs) if obs else None, "usage": g["usage"]})
    return steps


def steps_from_pi(trace):
    steps = []
    for i, s in enumerate(trace):
        calls = [{"id": c.get("id") or f"call_{i}_{j}", "name": c["name"], "arguments": _loads(c.get("arguments"))}
                 for j, c in enumerate(s["calls"])]
        steps.append({"step_number": i, "thought": s.get("thought") or None, "action": _action(calls),
                      "observation": "\n".join(s["observations"]) if s.get("observations") else None, "usage": s.get("usage")})
    return steps


def conversation(prompt, steps, final):
    """OpenAI-style messages, which AuditKIT uses to pair each call with its result."""
    msgs = [{"role": "user", "content": prompt}]
    for s in steps:
        a = s["action"]
        calls = [] if a is None else (a["tool_calls"] if "tool_calls" in a else [a])
        if calls:
            msgs.append({"role": "assistant", "content": s.get("thought") or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]})
            for c in calls:
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": s["observation"] or ""})
    msgs.append({"role": "assistant", "content": final})
    return msgs


def export(rows, tasks, harness, model, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {t["id"]: t for t in tasks}
    n_steps = 0
    with open(out / "trajectories.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            t = by_id[r["task"]]
            steps = steps_from_pi(r["trace"]) if harness == "pi" else steps_from_events(r.get("events", []), r.get("per_call", []), t["prompt"])
            final = next((x[1] for x in reversed(r.get("events", [])) if x[0] == "final"), None) if harness != "pi" else None
            final = final if final is not None else r["final"]
            n_calls = sum(0 if s["action"] is None else (len(s["action"]["tool_calls"]) if "tool_calls" in s["action"] else 1) for s in steps)
            n_steps += len(steps)
            rec = {"trajectory_id": f"{harness}-{r['task']}", "task": t["prompt"], "steps": steps, "final_response": final,
                   "reward": None,
                   "metadata": {"conversation": conversation(t["prompt"], steps, final), "tool_call_count": n_calls,
                                "question_id": r["task"], "level": r.get("level"), "own_verdict": r.get("correct"), "harness": harness, "model": model, "tier": r.get("tier"),
                                "seconds": r.get("seconds"),
                                "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                                          "total_tokens": r["tokens"], "embed_tokens": r.get("embed_tokens", 0), "model_calls": r["calls"]}}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "lexsi_provenance.json").write_text(json.dumps({
        "library": "leanloop", "version": "0.1.0", "base_model": model, "method": f"{harness} harness (no training)",
        "training_dataset": None, "note": "AgentTune-compatible trajectory file written by leanloop's exporter, not by AgentTune"},
        indent=1), encoding="utf-8")
    return len(rows), n_steps


def main(rows_path, tasks_path, harness, model, out_dir):
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8")]
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    n, s = export(rows, tasks, harness, model, out_dir)
    print(f"{n} trajectories, {s} steps -> {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:])
