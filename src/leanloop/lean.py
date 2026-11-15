"""The lean harness. THE ONLY FILE THE RESEARCH LOOP EDITS (like autoresearch's train.py).

X factor: a program-first cascade. Try the cheapest path first, check it with deterministic rules, and
only escalate when the check fails:

  tier 0  router (zero tokens to decide): small talk like "hey" gets one tiny chat call, no files, no program;
          a question about a file that does not exist is answered for free.
  tier R  questions about prose documents: micro-span pruning (prune.py), then ONE call over the relevant sentences.
  Safety (safety.py): tier-1 programs must be read-only (checked statically, never run otherwise); risky tier-2
          actions always ask; model code runs in an isolated child process with secrets removed from its environment.
  tier 1  ONE model call, no tool schemas. The prompt is the question, a file map, and evidence lines
          found by a plain keyword grep (zero model tokens to compute). The model writes one Python program;
          the harness runs it and the program's stdout IS the answer, so no extra "now answer" call is needed.
  tier 1b if the program crashes or prints something empty-looking, ONE repair call with the error.
  tier 2  only then: the full agent loop (tools + paging + dedupe + compaction), from experiments 1-10.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from . import llm, prune, safety
from .store import Store
from .tools import BASE_TOOLS, run_base_tool

# ----------------------------------------------------------------------------- shared helpers

MAP_CHARS = 1500     # cap on the workspace map
CODE_SUFFIXES = (".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".h")


def _oneline(text: str, n: int) -> str:
    return " ".join(text.split())[:n]


SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache",
             ".pytest_cache", "dist", "build", "target", "site-packages", ".idea", ".vscode"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".pdf", ".zip", ".gz", ".tar", ".bz2", ".xz",
                   ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyd", ".whl", ".mp3", ".mp4", ".mov",
                   ".avi", ".wav", ".ogg", ".woff", ".woff2", ".ttf", ".otf", ".class", ".jar", ".db", ".sqlite", ".parquet",
                   ".npy", ".npz", ".pkl", ".pt", ".pth", ".onnx", ".gguf", ".safetensors"}
MAX_FILES = 3000             # never scan more than this many files
MAX_READ_BYTES = 5_000_000   # evidence / prose scans read at most this much of any one file


def _files(root: Path):
    """Text-ish files under root, sorted, skipping hidden / vendor / build folders and known binary types."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() in BINARY_SUFFIXES:
                continue
            found.append(Path(dirpath) / name)
            if len(found) >= MAX_FILES:
                return sorted(found)
    return sorted(found)


def _is_text(p: Path) -> bool:
    """Cheap binary sniff: a NUL byte in the first 4 KB means binary."""
    try:
        with open(p, "rb") as f:
            return b"\0" not in f.read(4096)
    except OSError:
        return False


def _read_text(p: Path, limit: int = MAX_READ_BYTES) -> str:
    """Read at most `limit` bytes as UTF-8 text (invalid bytes replaced, CRLF -> LF like text mode would).
    Never loads a huge file whole."""
    try:
        with open(p, "rb") as f:
            return f.read(limit).decode("utf-8", errors="replace").replace("\r\n", "\n")
    except OSError:
        return ""


def _count_lines(p: Path) -> int:
    """Newline count, streamed in 1 MB chunks so a multi-GB log costs time but not memory."""
    n = 0
    try:
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                n += chunk.count(b"\n")
    except OSError:
        pass
    return n


def _workspace_map(root: Path) -> str:
    """Compact file map (like aider's repo-map): code files grouped by folder, data files with line count + first line."""
    code, data = {}, []
    for p in _files(root):
        rel = p.relative_to(root).as_posix()
        if p.suffix in CODE_SUFFIXES:
            code.setdefault(p.parent.relative_to(root).as_posix(), []).append(p.name)
            continue
        if not _is_text(p):
            continue
        head = _read_text(p, 4096).split("\n", 1)[0][:80]
        data.append(f"{rel} ({_count_lines(p)} lines): {head}")
    entries = data + [f"{d}/: {' '.join(fs)}" for d, fs in code.items()]
    out, used = [], 0
    for i, e in enumerate(entries):
        if used + len(e) + 1 > MAP_CHARS:          # cut on an entry boundary, and say how much is missing
            out.append(f"... +{len(entries) - i} more entries")
            break
        out.append(e)
        used += len(e) + 1
    return "\n".join(out)


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
    files = [p for p in _files(root) if _is_text(p)]
    names = {p.relative_to(root).as_posix() for p in files} | {p.name for p in files}
    q = question.lower()
    named = [p for p in files if p.relative_to(root).as_posix().lower() in q or p.name.lower() in q]
    kws = _keywords(question, names)
    if not kws:
        return ""
    rxs = [re.compile(r"(?<![\w])" + re.escape(k) + r"(?![\w])", re.I) for k in kws]
    docs = [(p.relative_to(root).as_posix(), _read_text(p).splitlines()) for p in (named or files)]
    # lines that read like instructions to a model are never offered as evidence (prompt-injection guard)
    hits = [[(f, i, ln) for f, lines in docs for i, ln in enumerate(lines, 1)
             if rx.search(ln) and not prune.INJECTION.search(safety.normalize(ln))] for rx in rxs]
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


CHILD_TIMEOUT = 30
MAX_PROMPT_CHARS = 3000      # a pasted wall of text must not overflow the 8k context
DECLINED = object()          # sentinel: the user said no to running the program


def _clip(text: str, n: int = MAX_PROMPT_CHARS) -> str:
    return text if len(text) <= n else text[:n] + " ...[truncated]"


def _ask(approve, kind: str, detail: str, risk: str | None = None) -> bool:
    """Ask for approval. Risky actions pass `risk` so the UI can insist on an explicit yes."""
    if not risk:
        return bool(approve(kind, detail))
    try:
        return bool(approve(kind, detail, risk))
    except TypeError:                      # a simple 2-argument approve(): fall back
        return bool(approve(kind, detail))


def _run_python(workspace: Path, code: str) -> tuple[int, str, str]:
    """Run model-written code in a child process: isolated interpreter (-I: workspace files cannot shadow the stdlib,
    no PYTHON* env, no user site), UTF-8 I/O whatever the Windows code page, no secrets in the environment,
    no stdin, and a time limit."""
    if "\n" not in code and "\\n" in code:  # small models often double-escape newlines in tool-call JSON
        code = code.replace("\\n", "\n").replace("\\t", "\t")
    try:
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code], cwd=workspace, env=safety.scrubbed_env(),
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=CHILD_TIMEOUT)
        return r.returncode, (r.stdout or "")[:200_000], (r.stderr or "")[:200_000]
    except subprocess.TimeoutExpired:
        return 1, "", f"error: timed out after {CHILD_TIMEOUT}s"
    except OSError as e:
        return 1, "", f"error: could not start python ({e})"


# ----------------------------------------------------------------------------- tier 1: program-first

PF_SYSTEM = ("Answer the question about the files in the current directory by writing ONE short Python 3 program "
             "that prints only the final answer (every name/value asked for, nothing else). "
             "Check the file formats in the map and evidence. Reply with just a ```python code block.")
PF_REPAIRS = 1
MAX_ANSWER_CHARS = 400
LONG_ANSWER_CAP = 1500
EMPTYISH = {"", "0", "0.0", "none", "nan", "[]", "{}", "()", "false", "null"}
ZEROS = {"0", "0.0"}


def _extract_code(content: str) -> str:
    """The program in a model reply: a closed ```python block, else an unclosed one (reply cut off), else bare code."""
    content = content or ""
    m = re.search(r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```", content, re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*)$", content, re.S | re.I)
    if m and m.group(1).strip("`\n ").strip():
        return m.group(1).rstrip("`\n ") + "\n"
    if "print(" in content:
        try:
            ast.parse(content)
            return content
        except SyntaxError:
            pass
    return ""


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


def _program_first(prompt: str, workspace: Path, model: str, chat, emit, approve):
    """Returns the answer, None (escalate to the agent loop), or DECLINED (the user refused to run the program)."""
    prompt = _clip(prompt)
    ev = _evidence(workspace, prompt)
    user = f"{prompt}\n\nFiles:\n{_workspace_map(workspace)}" + (f"\n\nEvidence:\n{ev}" if ev else "")
    messages = [{"role": "system", "content": PF_SYSTEM}, {"role": "user", "content": user}]
    zero_seen = False
    for attempt in range(PF_REPAIRS + 1):
        r = chat(model, messages, None, num_predict=600)
        content = r["message"].get("content", "") or ""
        code = _extract_code(content)
        emit("program", code or content)
        if not code:
            reason = "there was no ```python code block."
        elif (blocked := safety.unsafe_python(code)):
            emit("note", f"not run: the program {blocked}")      # questions are read-only; never run this, even if approved
            reason = f"it {blocked}. Questions are read-only: only read files and print the answer."
        elif not _ask(approve, "python", code):
            return DECLINED
        else:
            rc, out, err = _run_python(workspace, code)
            emit("output", (out + err).strip())
            reason, o = _verdict(rc, out, err), out.strip()
            if reason is None:
                return o
            if rc == 0 and o in ZEROS:
                if zero_seen:                      # the same zero from two independent attempts: believe it
                    return o
                zero_seen = True
            elif rc == 0 and len(o) > MAX_ANSWER_CHARS and attempt == PF_REPAIRS:
                return o[:LONG_ANSWER_CAP]         # a genuinely long answer (e.g. a list): keep it, don't escalate
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
            fn = c.get("function") if isinstance(c, dict) else None
            fn = fn if isinstance(fn, dict) else {}
            name, args = str(fn.get("name") or "unknown"), fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):         # a sloppy model may send a list or a bare string
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
                elif name in ("python", "bash", "write_file") and not _ask(
                        approve, name, json.dumps(args), _risk(name, args)):
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
    return f"I could not finish within {MAX_STEPS} steps."


def _risk(name: str, args: dict) -> str | None:
    """Why an agent-loop action needs an explicit yes even when 'run without asking' is on."""
    if name == "write_file":
        return "writes a file"
    if name == "python":
        return safety.unsafe_python(str(args.get("code", "")))
    if name == "bash":
        return safety.risky_command(str(args.get("command", "")))
    return None


# ----------------------------------------------------------------------------- entry point

CHAT_SYSTEM = ("You are leanloop, a coding agent that answers questions about the files in the current directory "
               "by writing small programs. This message needs no files: reply briefly, in one or two sentences.")
SMALL_TALK = re.compile(r"(hi|hey|hello|hiya|yo|sup|thanks|thank you|thx|ty|ok|okay|cool|nice|great|bye|goodbye|"
                        r"good (morning|afternoon|evening|night)|who are you|what are you|what can you do|help)\b[\s!.?]*$",
                        re.I)
DATA_WORDS = re.compile(r"\b(how many|count|sum|total|average|mean|max|min|list|find|which|where|show|file|files)\b", re.I)


def _stderr_emit(kind: str, text: str) -> None:
    print(f"    [{kind}] {_oneline(text, 200)}", file=sys.stderr)


def _is_chat(prompt: str, workspace: Path, ev: str) -> bool:
    """Tier-0 router (deterministic, zero tokens): small talk, or a very short message that names no file and
    matches nothing in the files, needs no program."""
    if SMALL_TALK.match(prompt.strip()):
        return True
    if ev or len(prompt.split()) > 3 or DATA_WORDS.search(prompt):
        return False
    files = list(_files(workspace))
    words = {w.strip(".,?!:;()'\"`").lower() for w in prompt.split()}
    stems = {p.stem.lower() for p in files} | {p.name.lower() for p in files}
    return not any(w and (w in stems or w.rstrip("s") in {s.rstrip("s") for s in stems}) for w in words)


PROSE = (".md", ".txt", ".rst")
DOC_SYSTEM = ("Answer the question using only the excerpts. Reply with the answer only, briefly. "
              "If the excerpts do not contain the answer, say so.")
NO_ANSWER = re.compile(r"(do(es)? not (contain|include|mention|specify|provide)|not (mentioned|found|specified|provided|available)"
                       r"|no (relevant )?information|cannot (find|determine))", re.I)
COUNT_SHAPE = re.compile(r"\b(how many|number of|count( of)?)\s+(lines|rows|words|characters|chars|bytes|columns|files)\b"
                         r"|\b(count|sum|average|total)\b.*\b(lines|rows|words)\b")
MAX_DOC_SPANS = 20000        # cap on spans embedded per question (large corpora need a persistent index, not this)
MAX_DOC_BYTES = 1_000_000    # read at most this much of any one document
_EMBED = prune.EmbedCache()   # shared across questions in a session: each document is embedded once


# Delimited rows ("| a | b |", tab-separated, consistent commas) are DATA even in a .txt/.md file: "which episode had the most
# viewers?" needs every row, so the idea was to treat it as a computation (write a program), not a retrieval (keep a few sentences).
# EXPERIMENTAL, OFF by default (LEANLOOP_TABLE_ROUTING=1 turns it on). On 40 FRESH documents it did NOT help: table accuracy
# 6/20 with it versus 8/20 without (p = 0.74), 3x the tokens, and the planted instruction was followed 4 times instead of 1.
TABLE_ROUTING = os.environ.get("LEANLOOP_TABLE_ROUTING") == "1"
# Experimental, OFF by default: warn when retrieved sentences give different numbers for the same thing (LEANLOOP_CONFLICT_CHECK=1).
CONFLICT_CHECK = os.environ.get("LEANLOOP_CONFLICT_CHECK") == "1"


def _looks_tabular(p: Path) -> bool:
    lines = [ln for ln in _read_text(p, 20000).splitlines() if ln.strip()][:30]
    if len(lines) < 3:
        return False
    shapes = []
    for ln in lines:
        for d in ("|", "\t", ","):
            if ln.count(d) >= 2:
                shapes.append((d, ln.count(d)))
                break
        else:
            shapes.append(None)
    found = [x for x in shapes if x]
    if len(found) < 0.6 * len(lines):
        return False
    shape = max(set(found), key=found.count)
    # most rows must share the same delimiter and column count; commas occur in all prose, so ask for more of them
    need = (0.8, 4) if shape[0] == "," else (0.6, 3)
    return len(lines) >= need[1] and found.count(shape) >= need[0] * len(lines)


def _is_prose_file(p: Path) -> bool:
    return p.suffix.lower() in PROSE and not (TABLE_ROUTING and _looks_tabular(p))


def _doc_route(prompt: str, workspace: Path, ev: str) -> bool:
    """Tier-R router (deterministic, zero tokens): is this a question about prose documents rather than data/code?"""
    files = list(_files(workspace))
    prose = [p for p in files if _is_prose_file(p)]
    if not prose:
        return False
    q = prompt.lower()
    others = [p for p in files if p not in prose]
    if any(p.name.lower() in q or p.relative_to(workspace).as_posix().lower() in q for p in others):
        return False
    if COUNT_SHAPE.search(q):
        return False                           # "how many lines/words/rows ..." is computed, not retrieved
    if any(p.name.lower() in q for p in prose):
        return True
    if not ev:
        return not others                      # nothing matched: documents-only workspace -> document tier
    sources = [line.split(":", 1)[0] for line in ev.splitlines()]
    return sum(Path(s).suffix.lower() in PROSE for s in sources) * 2 >= len(sources)


def _doc_answer(prompt: str, workspace: Path, model: str, chat, emit, embed) -> tuple[str, int] | None:
    """Tier R: micro-span pruning, then ONE model call over only the relevant sentences. None -> fall through."""
    spans = []
    for p in _files(workspace):
        if _is_prose_file(p) and _is_text(p):
            spans += prune.split_spans(_read_text(p, MAX_DOC_BYTES))
            if len(spans) >= MAX_DOC_SPANS:
                spans = spans[:MAX_DOC_SPANS]
                emit("note", f"only the first {MAX_DOC_SPANS} spans were searched")
                break
    if not spans:
        return None
    try:
        r = prune.prune(prompt, spans, embed=embed)
    except Exception as e:  # e.g. the embedding model is not pulled: fall back to the program-first path
        emit("note", f"embedding unavailable ({type(e).__name__}); skipping the document tier")
        return None
    emit("excerpts", f"{len(r.spans)} of {r.total_spans} spans kept, {r.dropped_guard} dropped by guard\n" + "\n".join(r.spans))
    for s in r.dropped[:3]:
        emit("note", f"guard dropped: {_oneline(s, 100)}")        # so a false positive is visible, not silent
    resp = chat(model, [{"role": "system", "content": DOC_SYSTEM},
                        {"role": "user", "content": "Excerpts:\n" + "\n".join(f"- {s}" for s in r.spans) + f"\n\nQuestion: {prompt}"}],
                None, num_predict=200)
    answer = (resp["message"].get("content") or "").strip()
    if CONFLICT_CHECK:       # opt-in: two relevant sentences that disagree on a number are shown to the user, not silently resolved
        gone = set(r.dropped)
        for a, b in prune.find_conflicts(r.spans, pool=[x for x in spans if x not in gone])[:2]:
            emit("note", f"the documents disagree: {_oneline(a, 90)}  <>  {_oneline(b, 90)}")
            answer += f"\n(Warning: the documents contain conflicting statements: \"{_oneline(a, 140)}\" versus \"{_oneline(b, 140)}\".)"
    return answer, r.embed_tokens


FILE_REF = re.compile(r"(?<![\w/.\\-])([\w./\\-]+\.(?:csv|tsv|log|txt|md|rst|jsonl?|ya?ml|toml|ini|cfg|conf|xml|html?|sql"
                      r"|py|js|ts|rs|go|java|cpp|c|h))\b", re.I)


def _missing_files(prompt: str, workspace: Path) -> list[str]:
    """File names the question mentions that do not exist here. Free to decide, and it saves a whole agent loop
    that could only discover 'not found'. Skipped when the question contains a URL (the file may be remote)."""
    if "://" in prompt:
        return []
    rels = [p.relative_to(workspace).as_posix().lower() for p in _files(workspace)]
    # Mask every file that DOES exist first, longest names first. A name with spaces or symbols
    # ("beauty and the labor market.csv", "DES=+2006261.csv") would otherwise be misread as the fragment
    # "market.csv" / "2006261.csv" and reported as missing.
    text = prompt
    for known in sorted({Path(r).name for r in rels} | set(rels), key=len, reverse=True):
        # only a name that stands alone: "sales.csv" inside "old/sales.csv" is a different (maybe missing) path
        text = re.sub(r"(?<![\w/\\.\-])" + re.escape(known) + r"(?!\w)", " ", text, flags=re.IGNORECASE)
    out = []
    for ref in dict.fromkeys(m.group(1).strip(".") for m in FILE_REF.finditer(text)):
        norm = ref.replace("\\", "/").lstrip("./").lower()
        # exists if some file's path is exactly this, or ends with it ("billing/tax.py" finds "src/billing/tax.py"),
        # or it is the tail of a longer existing file name ("market.csv" in "labor market.csv")
        if (not any(r == norm or r.endswith("/" + norm) or r.endswith(norm) for r in rels)
                and not (workspace / ref).exists()):
            out.append(ref)
    return out


def run(prompt: str, workspace: Path, model: str, *, chat=llm.chat, emit=_stderr_emit,
        approve=lambda kind, detail: True, embed=None) -> dict:
    """emit(kind, text) reports progress (the CLI draws it); approve(kind, detail[, risk]) can veto running code.
    Result: {"final", "tier", "embed_tokens"}; embedding tokens are reported apart from the model's own tokens."""
    workspace = Path(workspace)
    if not workspace.is_dir():
        raise NotADirectoryError(f"not a folder: {workspace}")
    prompt = (prompt or "").strip()
    if not prompt:
        return {"final": "Ask me a question about the files in this folder.", "tier": 0, "embed_tokens": 0}
    prompt = _clip(prompt)
    missing = _missing_files(prompt, workspace)
    if missing:
        names = [p.relative_to(workspace).as_posix() for p in _files(workspace)]
        shown = (", ".join(names[:8]) + (f" (+{len(names) - 8} more)" if len(names) > 8 else "")) if names else "none"
        final = f"There is no file named {', '.join(repr(m) for m in missing)} in this folder. Files here: {shown}."
        emit("tier", "0: no such file")
        emit("final", final)
        return {"final": final, "tier": 0, "embed_tokens": 0}
    ev = _evidence(workspace, prompt)
    if _is_chat(prompt, workspace, ev):
        emit("tier", "0: chat")
        r = chat(model, [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": prompt}], None,
                 num_predict=120)
        final = (r["message"].get("content") or "").strip()
        emit("final", final)
        return {"final": final, "tier": 0, "embed_tokens": 0}
    blocked: list[str] = []
    user_emit = emit

    def emit(kind, text):                      # noqa: F811  (wraps the caller's emit to remember blocked programs)
        if kind == "note" and text.startswith("not run: the program "):
            blocked.append(text[len("not run: the program "):])
        user_emit(kind, text)
    embed_used = 0
    if _doc_route(prompt, workspace, ev):
        emit("tier", "R: document retrieval")
        got = _doc_answer(prompt, workspace, model, chat, emit, embed or _EMBED)
        if got is not None:
            embed_used = got[1]
            if got[0] and not NO_ANSWER.search(got[0]):
                emit("final", got[0])
                return {"final": got[0], "tier": "R", "embed_tokens": embed_used}
            emit("note", "the excerpts did not answer it; trying the program path")
    emit("tier", "1: program-first")
    final = _program_first(prompt, workspace, model, chat, emit, approve)
    tier = 1
    if final is DECLINED:
        final = "Cancelled: you declined to run the program."
    elif final is None:
        emit("tier", "2: agent loop")
        final, tier = _agent_loop(prompt, workspace, model, chat, emit, approve), 2
    if blocked and final and not final.startswith("Cancelled"):
        final = (f"Blocked: a program that {blocked[0]} was not run. leanloop answers questions by reading files; "
                 f"it does not change files or use the network.\nWhat I could answer by reading:\n{final}")
    emit("final", final)
    return {"final": final, "tier": tier, "embed_tokens": embed_used}
