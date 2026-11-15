"""Deterministic safety checks (zero model tokens) for model-written code and for text that comes from files.

These are a best-effort safety net, NOT a sandbox: a determined attacker or a very creative model can get around
static checks. They exist to (a) stop the common accidents (a Q&A program that deletes or uploads something),
(b) force a human confirmation for anything risky, and (c) keep secrets out of the child process.
Real isolation needs an OS-level sandbox (container, restricted user), which is out of scope here.
"""

from __future__ import annotations

import ast
import os
import re
import unicodedata

# ----------------------------------------------------------------------------- child-process environment

SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE)", re.I)


def scrubbed_env(env: dict | None = None) -> dict:
    """Environment for model-written code: everything except variables whose name looks like a secret."""
    return {k: v for k, v in (os.environ if env is None else env).items() if not SECRET_NAME.search(k)}


# ----------------------------------------------------------------------------- text from files

_INVISIBLE = {c: None for c in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E)}


def normalize(text: str) -> str:
    """NFKC-fold and strip zero-width / bidi characters so look-alike or invisible-character tricks can't dodge a regex."""
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE)


# ----------------------------------------------------------------------------- python: read-only check

BAD_MODULES = {"subprocess", "socket", "shutil", "ctypes", "requests", "httpx", "urllib.request", "urllib3",
               "http.client", "http.server", "ftplib", "smtplib", "telnetlib", "multiprocessing", "webbrowser",
               "pty", "xmlrpc", "paramiko", "pexpect"}
OS_DANGEROUS = {"remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace", "system", "popen", "startfile",
                "kill", "killpg", "chmod", "chown", "truncate", "mkdir", "makedirs", "symlink", "link", "putenv",
                "execv", "execve", "execl", "execlp", "execvp", "spawnl", "spawnv", "fork", "chdir"}
PATH_METHODS = {"unlink", "rmdir", "write_text", "write_bytes", "mkdir", "touch", "symlink_to", "hardlink_to", "chmod"}
WRITER_METHODS = {"to_csv", "to_excel", "to_json", "to_pickle", "to_parquet", "to_sql", "to_feather", "savetxt", "dump"}
BAD_BUILTINS = {"eval", "exec", "compile", "__import__"}
BAD_DUNDERS = {"__builtins__", "__subclasses__", "__globals__", "__loader__", "__code__", "__mro__", "__bases__"}


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _bad_module(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in BAD_MODULES)


def unsafe_python(code: str) -> str | None:
    """Why this program is not read-only (writes/deletes files, uses the network or a subprocess, evals), else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None                       # let it fail normally; the error goes back to the model
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _bad_module(a.name):
                    return f"imports {a.name} (network / processes / file operations)"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _bad_module(mod) or any(_bad_module(f"{mod}.{a.name}") for a in node.names):
                return f"imports from {mod} (network / processes / file operations)"
            if mod == "os" and any(a.name in OS_DANGEROUS for a in node.names):
                return "imports a dangerous os function"
        elif isinstance(node, ast.Attribute) and node.attr in BAD_DUNDERS:
            return f"touches {node.attr}"
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            last = name.rsplit(".", 1)[-1]
            if name in BAD_BUILTINS:
                return f"calls {name}()"
            if name.startswith("os.") and last in OS_DANGEROUS:
                return f"calls {name} (changes files or runs commands)"
            attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if attr in PATH_METHODS:
                return f"calls .{attr}() (changes files)"
            if attr in WRITER_METHODS and (node.args or node.keywords):
                return f"calls .{attr}() with a target (writes a file)"
            if last == "open":
                mode = None
                # builtin open(file, mode) / io.open / gzip.open take the mode second; Path(...).open(mode) takes it first
                idx = 0 if isinstance(node.func, ast.Attribute) and _dotted(node.func.value) not in (
                    "io", "codecs", "os", "gzip", "bz2", "lzma", "tarfile", "zipfile") else 1
                if len(node.args) > idx and isinstance(node.args[idx], ast.Constant):
                    mode = node.args[idx].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if isinstance(mode, str) and set(mode) & set("wax+"):
                    return f"opens a file for writing (mode {mode!r})"
    return None


# ----------------------------------------------------------------------------- bash: risk check

RISKY_SHELL = re.compile(
    r"(^|[\s;&|(])(rm|rmdir|del|erase|mv|move|cp|copy|chmod|chown|dd|mkfs|format|sudo|kill|killall|shutdown|reboot|"
    r"curl|wget|nc|ncat|ssh|scp|sftp|ftp|tee|truncate|ln|mkdir|touch)(\s|$)"
    r"|[^<>0-9&]>{1,2}\s*[^&\s/]|\bsed\s+-i\b|\bgit\s+(push|reset|clean|checkout|rebase)\b|\b(pip|npm|apt|brew)\s+install\b"
    r"|\|\s*(sh|bash|zsh|python\d?)\b|\beval\b|:\(\)\s*\{", re.I)


def risky_command(cmd: str) -> str | None:
    """Why this shell command needs an explicit yes (deletes/moves/writes files, network, installs), else None."""
    m = RISKY_SHELL.search(cmd or "")
    return f"shell command may change files, use the network or install software ({m.group(0).strip()[:20]!r})" if m else None
