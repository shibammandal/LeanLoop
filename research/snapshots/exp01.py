"""The lean harness. THE ONLY FILE THE RESEARCH LOOP EDITS (like autoresearch's train.py).

Starts as an exact copy of the lean mode in agent.py. Paging: every tool output is saved to a Store;
the prompt keeps a small working set (receipts for big outputs, eviction of old outputs, dedupe).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import llm
from .store import Store
from .tools import LEAN_TOOLS, run_base_tool

SYSTEM = ("You are a coding agent working in the current directory. Before parsing a file, look at its first "
          "lines (head -3) to learn its format; match exact fields, not loose substrings. Find files with ls or "
          "grep -rl instead of guessing names. Keep using tools until you have the answer; never reply with a plan. "
          "Final reply: the answer only, briefly. Long outputs are stored as [#N ...]; read more with "
          "fetch(handle,start,end) or search(handle,pattern).")
TOOLS = LEAN_TOOLS

MAX_STEPS = 12
LEAN_INLINE = 600    # outputs up to this many chars are shown whole
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


def run(prompt: str, workspace: Path, model: str, *, chat=llm.chat) -> dict:
    store, seen = Store(), {}
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    final = ""

    for step in range(1, MAX_STEPS + 1):
        idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
        old = set(idx[:-KEEP_FULL]) if len(idx) > KEEP_FULL else set()
        send = [dict(m, content=_evicted(m["_h"], m["_raw"])) if i in old else m for i, m in enumerate(messages)]
        r = chat(model, _clean(send), TOOLS)
        msg = r["message"]
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            final = msg.get("content", "")
            break
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            print(f"    [{step}] {name} {json.dumps(args)[:200]}", file=sys.stderr)
            key = (name, json.dumps(args, sort_keys=True))
            if name in ("read_file", "bash") and key in seen:
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
                shown = raw if len(raw) <= LEAN_INLINE else _receipt(h, raw)
                seen.setdefault(key, (step, h))
            messages.append({"role": "tool", "tool_name": name, "content": shown, "_h": h, "_raw": raw})

    return {"final": final}
