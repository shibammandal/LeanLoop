"""The agent loop, in two modes that differ only in how they manage the prompt.

naive: what most harnesses do. Every tool output stays in the prompt in full and is re-sent every step.
lean:  paging. Everything is saved to a Store. The prompt keeps a small working set:
         - big outputs show a short receipt + preview, with a handle to fetch/search the rest
         - outputs older than the last KEEP_FULL are evicted to a one-line summary
         - an identical repeated read/bash call is answered with a pointer, not re-sent
"""

from __future__ import annotations

import json
from pathlib import Path

from . import llm
from .store import Store
from .tools import BASE_TOOLS, LEAN_TOOLS, run_base_tool

SYS_NAIVE = ("You are a coding agent. Work in the current directory using the tools. "
             "When finished, reply with the final answer only, briefly.")
SYS_LEAN = (SYS_NAIVE + " Long outputs are stored and shown as [#N ...]; use fetch(handle,start,end) or "
            "search(handle,pattern) to read more. Prefer commands that print only what you need.")

NAIVE_CAP = 6000     # chars of one tool output kept in the naive prompt (needed to fit an 8k context)
LEAN_INLINE = 600    # outputs up to this many chars are shown whole in lean mode
PREVIEW = 300        # chars of preview for bigger outputs
KEEP_FULL = 2        # most recent tool outputs kept as shown; older ones are evicted


def _oneline(text: str, n: int) -> str:
    return " ".join(text.split())[:n]


def _receipt(h: int, text: str) -> str:
    lines = text.count("\n") + 1
    return f"[#{h}: {lines} lines, {len(text)} chars. Preview: {_oneline(text, PREVIEW)} ...] Use fetch/search to read more."


def _evicted(h: int, text: str) -> str:
    return f"[#{h} evicted: {_oneline(text, 80)} ...] Use fetch/search to re-read."


def _clean(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def run(prompt: str, workspace: Path, mode: str, model: str, *, max_steps: int = 12, num_ctx: int = 8192,
        chat=llm.chat) -> dict:
    lean = mode == "lean"
    store, seen = Store(), {}
    tools = LEAN_TOOLS if lean else BASE_TOOLS
    if mode == "naivemap":   # the naive loop (full history, same tools) but told which files exist: isolates the cascade's contribution
        from .lean import _workspace_map
        prompt = f"{prompt}\n\nFiles:\n{_workspace_map(workspace)}"
    messages = [{"role": "system", "content": SYS_LEAN if lean else SYS_NAIVE}, {"role": "user", "content": prompt}]
    per_call, trace, final, stopped = [], [], "", "budget"

    for step in range(1, max_steps + 1):
        send = messages
        if lean:  # evict all but the most recent KEEP_FULL tool outputs
            idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
            old = set(idx[:-KEEP_FULL]) if len(idx) > KEEP_FULL else set()
            send = [dict(m, content=_evicted(m["_h"], m["_raw"])) if i in old else m for i, m in enumerate(messages)]
        r = chat(model, _clean(send), tools, num_ctx=num_ctx)
        per_call.append({"step": step, "prompt": r["prompt_tokens"], "completion": r["completion_tokens"]})
        msg = r["message"]
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            final, stopped = msg.get("content", ""), "answer"
            break
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            key = (name, json.dumps(args, sort_keys=True))
            if lean and name in ("read_file", "bash") and key in seen:
                k_step, k_h = seen[key]
                raw, shown = f"(repeat of step {k_step})", f"[duplicate of step {k_step}; result unchanged, see #{k_h}]"
                h = store.put(raw)
            else:
                if name == "fetch":
                    raw = store.fetch(int(args.get("handle", 0)), args.get("start") or 1, args.get("end"))
                elif name == "search":
                    raw = store.search(int(args.get("handle", 0)), str(args.get("pattern", "")))
                else:
                    raw = run_base_tool(workspace, name, args)
                h = store.put(raw)
                if lean:
                    shown = raw if len(raw) <= LEAN_INLINE else _receipt(h, raw)
                    seen.setdefault(key, (step, h))
                else:
                    shown = raw if len(raw) <= NAIVE_CAP else raw[:NAIVE_CAP] + f"\n[truncated, {len(raw)} chars total]"
            trace.append({"step": step, "name": name, "args": args})
            messages.append({"role": "tool", "tool_name": name, "content": shown, "_h": h, "_raw": raw})

    p = sum(c["prompt"] for c in per_call)
    o = sum(c["completion"] for c in per_call)
    return {"mode": mode, "final": final, "stopped": stopped, "steps": len(per_call), "prompt_tokens": p,
            "completion_tokens": o, "total_tokens": p + o, "per_call": per_call, "trace": trace,
            "context_overflow": any(c["prompt"] >= num_ctx - 16 for c in per_call)}
