"""The lean harness. THE ONLY FILE THE RESEARCH LOOP EDITS (like autoresearch's train.py).

X factor: a program-first cascade. Try the cheapest path first, check it with deterministic rules, and
only escalate when the check fails:

  tier 1  ONE model call, no tool schemas. The prompt is the question, a file map, and evidence lines
          found by a plain keyword grep (zero model tokens to compute). The model writes one Python program;
          the harness runs it and the program's stdout IS the answer, so no extra "now answer" call is needed.
  tier 1b if the program crashes or prints something empty-looking, ONE repair call with the error.
  tier 2  only then: the full agent loop (tools + paging + dedupe + compaction), from experiments 1-10.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

from . import llm
from .store import Store
from .tools import BASE_TOOLS, run_base_tool

# ----------------------------------------------------------------------------- shared helpers

MAP_CHARS = 1500     # cap on the workspace map
CODE_SUFFIXES = (".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".h")


def _oneline(text: str, n: int) -> str:
    return " ".join(text.split())[:n]


def _files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."):
            yield p


def _workspace_map(root: Path) -> str:
    """Compact file map (like aider's repo-map): code files grouped by folder, data files with line count + first line."""
    code, data = {}, []
    for p in _files(root):
        rel = p.relative_to(root).as_posix()
        if p.suffix in CODE_SUFFIXES:
            code.setdefault(p.parent.relative_to(root).as_posix(), []).append(p.name)
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        data.append(f"{rel} ({text.count(chr(10))} lines): {text.split(chr(10), 1)[0][:80]}")
    out = data + [f"{d}/: {' '.join(fs)}" for d, fs in code.items()]
    return "\n".join(out)[:MAP_CHARS]


STOP = set("""what which who how many much is are was were the a an of in on for to and or with from by at as
that this these those does do did has have had its it their there be been value values default file files line lines
number numbers total most least largest smallest highest lowest first last reply name answer only give find show
tell me please times count""".split())
EVIDENCE_PER_FILE, EVIDENCE_MAX = 3, 6


def _keywords(question: str, file_names: set[str]) -> list[str]:
    words = [w.strip(".,?!:;()'\"`") for w in question.split()]
    return list(dict.fromkeys(w for w in words if len(w) >= 3 and w.lower() not in STOP
                              and w not in file_names and not any(w.endswith(f) for f in file_names)))


def _evidence(root: Path, question: str) -> str:
    """Deterministic retrieval: lines that match the most question keywords (word-boundary, case-insensitive).
    If the question names files, only those files are scanned."""
    files = [p for p in _files(root) if p.stat().st_size < 5_000_000]
    names = {p.relative_to(root).as_posix() for p in files} | {p.name for p in files}
    named = [p for p in files if p.relative_to(root).as_posix() in question or p.name in question]
    kws = _keywords(question, names)
    if not kws:
        return ""
    rxs = [re.compile(r"(?<![\w])" + re.escape(k) + r"(?![\w])", re.I) for k in kws]
    docs = []
    for p in named or files:
        try:
            docs.append((p.relative_to(root).as_posix(), p.read_text(encoding="utf-8", errors="replace").splitlines()))
        except OSError:
            continue
    hits = [[(f, i, ln) for f, lines in docs for i, ln in enumerate(lines, 1) if rx.search(ln)] for rx in rxs]
    # IDF-style weights: identifiers (digits, _, =, ALLCAPS) count more; rare keywords count more than common ones
    weight = [(3.0 if re.search(r"[\d_=]", k) or (k.isupper() and len(k) > 2) else 1.0) / (1 + math.log10(1 + len(h)))
              for k, h in zip(kws, hits)]
    score: dict[tuple, float] = {}
    for w, h in zip(weight, hits):
        for key in h:
            score[key] = score.get(key, 0.0) + w
    ranked, per_file = [], {}
    for (f, i, ln), _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        if per_file.get(f, 0) < EVIDENCE_PER_FILE:
            per_file[f] = per_file.get(f, 0) + 1
            ranked.append((f, i, ln))
        if len(ranked) == EVIDENCE_MAX:
            break
    return "\n".join(f"{f}:{i}: {ln[:120]}" for f, i, ln in sorted(ranked))


def _run_python(workspace: Path, code: str) -> tuple[int, str, str]:
    if "\n" not in code and "\\n" in code:  # small models often double-escape newlines in tool-call JSON
        code = code.replace("\\n", "\n").replace("\\t", "\t")
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=workspace, capture_output=True, text=True,
                           timeout=30, errors="replace")
        return r.returncode, (r.stdout or "")[:200_000], (r.stderr or "")[:200_000]
    except subprocess.TimeoutExpired:
        return 1, "", "error: timed out after 30s"


# ----------------------------------------------------------------------------- tier 1: program-first

PF_SYSTEM = ("Answer the question about the files in the current directory by writing ONE short Python 3 program "
             "that prints only the final answer (every name/value asked for, nothing else). "
             "Check the file formats in the map and evidence. Reply with just a ```python code block.")
PF_REPAIRS = 1
MAX_ANSWER_CHARS = 400
EMPTYISH = {"", "0", "0.0", "none", "nan", "[]", "{}", "()", "false", "null"}


def _extract_code(content: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", content or "", re.S)
    return m.group(1) if m else ""


def _verdict(rc: int, out: str, err: str) -> str | None:
    """None if the run looks like a real answer, else a short reason to send back to the model."""
    if rc != 0:
        return "it failed:\n" + err.strip()[-600:]
    o = out.strip()
    if o.lower().strip("'\"") in EMPTYISH:
        return f"it printed {o!r}, which looks empty or wrong. Check the exact file format (see the evidence lines)."
    if len(o) > MAX_ANSWER_CHARS:
        return f"it printed {len(o)} chars; print only the final answer."
    return None


def _program_first(prompt: str, workspace: Path, model: str, chat, emit, approve) -> str | None:
    ev = _evidence(workspace, prompt)
    user = f"{prompt}\n\nFiles:\n{_workspace_map(workspace)}" + (f"\n\nEvidence:\n{ev}" if ev else "")
    messages = [{"role": "system", "content": PF_SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(PF_REPAIRS + 1):
        r = chat(model, messages, None, num_predict=600)
        content = r["message"].get("content", "") or ""
        code = _extract_code(content)
        emit("program", code or content)
        if not code:
            reason = "there was no ```python code block."
        elif not approve("python", code):
            return None
        else:
            rc, out, err = _run_python(workspace, code)
            emit("output", (out + err).strip())
            reason = _verdict(rc, out, err)
            if reason is None:
                return out.strip()
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": f"The program did not work: {reason}\nReply with a fixed ```python block."}]
    return None


# ----------------------------------------------------------------------------- tier 2: agent loop (exp 1-10)

SYSTEM = ("You are a coding agent. Work in the current directory using the tools. "
          "When finished, reply with the final answer only, briefly."
          " Long outputs are stored and shown as [#N ...]; use view(handle, start, end) or "
          "view(handle, pattern) to read more. Prefer commands that print only what you need. "
          "For parsing or counting data, prefer the python tool.")


def _tool(name: str, desc: str, required: list[str], **props: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": required}}}


TOOLS = BASE_TOOLS + [
    _tool("view", "Show lines start-end of stored output #handle, or its lines matching regex pattern.",
          ["handle"], handle="integer", pattern="string", start="integer", end="integer"),
    _tool("python", "Run Python code in the workspace; print() what you need.", ["code"], code="string"),
]

MAX_STEPS = 12
LEAN_INLINE = 600    # outputs up to this many chars are shown whole
PREVIEW = 300        # chars of preview for bigger outputs
KEEP_TURNS = 1       # most recent model turns kept verbatim; older turns are compacted into one note


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


def _agent_loop(prompt: str, workspace: Path, model: str, chat, emit, approve) -> str:
    store, seen = Store(), {}
    user = f"{prompt}\n\nFiles:\n{_workspace_map(workspace)}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    for step in range(1, MAX_STEPS + 1):
        r = chat(model, _clean(_compact(messages, KEEP_TURNS)), TOOLS)
        msg = r["message"]
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", "") or ""
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            emit("call", f"{name} {json.dumps(args)[:200]}")
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
                elif name in ("python", "bash", "write_file") and not approve(name, json.dumps(args)):
                    raw = "error: the user denied this action"
                elif name == "python":
                    rc, out, err = _run_python(workspace, str(args.get("code", "")))
                    raw = (out + err) or "(no output)"
                else:
                    raw = run_base_tool(workspace, name, args)
                h = store.put(raw)
                shown = raw if len(raw) <= LEAN_INLINE else _receipt(h, raw)
                seen.setdefault(key, (step, h))
            emit("output", shown)
            messages.append({"role": "tool", "tool_name": name, "content": shown, "_h": h, "_raw": raw,
                             "_call": f"{name}({_oneline(json.dumps(args), 60)})"})
    return ""


# ----------------------------------------------------------------------------- entry point

def _stderr_emit(kind: str, text: str) -> None:
    print(f"    [{kind}] {_oneline(text, 200)}", file=sys.stderr)


def run(prompt: str, workspace: Path, model: str, *, chat=llm.chat, emit=_stderr_emit,
        approve=lambda kind, detail: True) -> dict:
    """emit(kind, text) reports progress (the CLI draws it); approve(kind, detail) can veto running code."""
    emit("tier", "1: program-first")
    final = _program_first(prompt, workspace, model, chat, emit, approve)
    tier = 1
    if final is None:
        emit("tier", "2: agent loop")
        final, tier = _agent_loop(prompt, workspace, model, chat, emit, approve), 2
    emit("final", final)
    return {"final": final, "tier": tier}
