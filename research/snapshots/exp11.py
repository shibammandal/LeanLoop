"""The lean harness. THE ONLY FILE THE RESEARCH LOOP EDITS (like autoresearch's train.py).

Starts as an exact copy of the lean mode in agent.py. Paging: every tool output is saved to a Store;
the prompt keeps a small working set (receipts for big outputs, eviction of old outputs, dedupe).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import llm
from .store import Store
from .tools import BASE_TOOLS, run_base_tool

SYSTEM = ("Coding agent in the current directory. Big outputs appear as [#N ...]; read them with view. "
          "Prefer python for parsing or counting; print only what you need. Final reply: the answer only.")


def _tool(name: str, desc: str, required: list[str], **props: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": required}}}


TOOLS = BASE_TOOLS + [
    _tool("view", "Show lines start-end of stored output #handle, or its lines matching regex pattern.",
          ["handle"], handle="integer", pattern="string", start="integer", end="integer"),
    _tool("python", "Run Python code in the workspace; print() what you need.", ["code"], code="string"),
]


def _run_python(workspace: Path, code: str) -> str:
    if "\n" not in code and "\\n" in code:  # small models often double-escape newlines in tool-call JSON
        code = code.replace("\\n", "\n").replace("\\t", "\t")
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=workspace, capture_output=True, text=True,
                           timeout=30, errors="replace")
        return ((r.stdout or "") + (r.stderr or ""))[:200_000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "error: timed out"

MAX_STEPS = 12
LEAN_INLINE = 600    # outputs up to this many chars are shown whole
PREVIEW = 300        # chars of preview for bigger outputs
KEEP_TURNS = 1       # most recent model turns kept verbatim; older turns are compacted into one note


def _oneline(text: str, n: int) -> str:
    return " ".join(text.split())[:n]


def _receipt(h: int, text: str) -> str:
    lines = text.count("\n") + 1
    return f"[#{h}: {lines} lines, {len(text)} chars. Preview: {_oneline(text, PREVIEW)} ...] Use view to read more."


def _int(v, default):
    digits = "".join(ch for ch in str(v) if ch.isdigit()) if v is not None else ""
    return int(digits) if digits else default


def _compact(messages: list[dict], keep: int) -> list[dict]:
    """Compaction: keep the system+task messages and the last `keep` turns; squash older turns
    (the assistant's call + its tool results) into one short note with handles to re-read them."""
    head, turns = messages[:2], []
    for m in messages[2:]:
        if m["role"] == "assistant" or not turns:
            turns.append([])
        turns[-1].append(m)
    if len(turns) <= keep:
        return messages
    notes = [f"- {m['_call']} -> {_oneline(m['_raw'], 100)} (#{m['_h']})"
             for t in turns[:-keep] for m in t if m["role"] == "tool"]
    digest = {"role": "user", "content": "Earlier steps (compacted; use view to re-read):\n" + "\n".join(notes)}
    return head + [digest] + [m for t in turns[-keep:] for m in t]


def _clean(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


MAP_CHARS = 1500     # cap on the workspace map


def _workspace_map(root: Path) -> str:
    """Compact file map (like aider's repo-map): code files grouped by folder, data files with line count + first line."""
    code, data = {}, []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name.startswith("."):
            continue
        rel = p.relative_to(root).as_posix()
        if p.suffix in (".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".h"):
            code.setdefault(p.parent.relative_to(root).as_posix(), []).append(p.name)
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        first = text.split("\n", 1)[0][:80]
        data.append(f"{rel} ({text.count(chr(10))} lines): {first}")
    out = data + [f"{d}/: {' '.join(fs)}" for d, fs in code.items()]
    return "\n".join(out)[:MAP_CHARS]


def run(prompt: str, workspace: Path, model: str, *, chat=llm.chat) -> dict:
    store, seen = Store(), {}
    user = f"{prompt}\n\nFiles:\n{_workspace_map(workspace)}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    final = ""

    for step in range(1, MAX_STEPS + 1):
        send = _compact(messages, KEEP_TURNS)
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
            if name in ("read_file", "bash", "python") and key in seen:
                k_step, k_h = seen[key]
                raw, shown = f"(repeat of step {k_step})", f"[duplicate of step {k_step}; result unchanged, see #{k_h}]"
                h = store.put(raw)
            else:
                if name in ("view", "fetch", "search"):
                    hd = _int(args.get("handle"), 0)
                    if args.get("pattern"):
                        raw = store.search(hd, str(args["pattern"]))
                    else:
                        raw = store.fetch(hd, _int(args.get("start"), 1), _int(args.get("end"), None))
                elif name == "python":
                    raw = _run_python(workspace, str(args.get("code", "")))
                else:
                    raw = run_base_tool(workspace, name, args)
                h = store.put(raw)
                shown = raw if len(raw) <= LEAN_INLINE else _receipt(h, raw)
                seen.setdefault(key, (step, h))
            messages.append({"role": "tool", "tool_name": name, "content": shown, "_h": h, "_raw": raw,
                             "_call": f"{name}({_oneline(json.dumps(args), 60)})"})

    return {"final": final}
