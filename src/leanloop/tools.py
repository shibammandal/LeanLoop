"""Tools the agent can call. Runs inside a throwaway workspace directory."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import safety

BASE_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command in the workspace.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]

LEAN_TOOLS = BASE_TOOLS + [
    {"type": "function", "function": {
        "name": "fetch", "description": "Show lines of a stored output by handle number.",
        "parameters": {"type": "object", "properties": {"handle": {"type": "integer"}, "start": {"type": "integer"},
                                                        "end": {"type": "integer"}}, "required": ["handle"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Regex-search a stored output by handle number.",
        "parameters": {"type": "object", "properties": {"handle": {"type": "integer"}, "pattern": {"type": "string"}},
                       "required": ["handle", "pattern"]}}},
]

HARD_CAP = 200_000


def _safe(workspace: Path, rel: str) -> Path | None:
    p = (workspace / rel).resolve()
    return p if workspace.resolve() in p.parents or p == workspace.resolve() else None


def run_base_tool(workspace: Path, name: str, args: dict) -> str:
    try:
        if name == "read_file":
            p = _safe(workspace, str(args.get("path", "")))
            if p is None or not p.is_file():
                return f"error: cannot read {args.get('path')}"
            with open(p, "rb") as f:                      # never load a huge file whole
                return f.read(HARD_CAP).decode("utf-8", errors="replace")
        if name == "write_file":
            p = _safe(workspace, str(args.get("path", "")))
            if p is None:
                return "error: path outside workspace"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {p.name}"
        if name == "bash":
            r = subprocess.run(["bash", "-c", str(args.get("command", ""))], cwd=workspace, env=safety.scrubbed_env(),
                               stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=30)
            return ((r.stdout or "") + (r.stderr or ""))[:HARD_CAP] or "(no output)"
    except subprocess.TimeoutExpired:
        return "error: command timed out"
    except Exception as e:  # tool errors go back to the model as text
        return f"error: {e}"
    return f"error: unknown tool {name}"
