"""leanloop CLI: an interactive terminal front end for the lean harness (our own code, built on rich + prompt_toolkit).

    leanloop                      interactive session in the current directory
    leanloop -p "question"        one-shot: print the answer and exit
    options: --model NAME  --cwd DIR  --yes (run code without asking)

Layout (in the spirit of Claude Code's terminal UI, written from scratch): a rounded welcome box with a small
infinity logo, the conversation (● steps, ⎿ outputs), a rounded input box with a > prompt, and "? for shortcuts"
plus a token meter underneath. Model-written code runs in YOUR directory, so by default you are asked first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import lean, llm, prune

VERSION = "0.1.0"
IN_PRICE, OUT_PRICE = 3 / 1e6, 15 / 1e6   # illustrative reference API price, not a quote

PURPLE = "#A855F7"
ACCENT = PURPLE
BORDER = "#6B4FA0"
DIM = "#8A84A3"
ERR = "#FF5C8A"
CODE_BG = "default"   # background of shown programs (the terminal's own)

PT_STYLE = {"border": BORDER, "arrow": f"bold {ACCENT}", "placeholder": "#5E5878",
            "bottom-toolbar": f"noreverse bg:default {DIM}", "bottom-toolbar.text": f"noreverse bg:default {DIM}"}

LOGO = "\n".join([
    r"  -====:  :====-  ",
    r" =*:--:*==*:--:*= ",
    r" =*:--:*==*:--:*= ",
    r"  -====:  :====-  ",
])

HELP = f"""[bold]?[/] or [bold]/help[/]   this help
[bold]/tokens[/]        session token totals
[bold]/model NAME[/]    switch the Ollama model
[bold]/yes[/]           toggle asking before code runs
[bold]/clear[/]         reset session totals
[bold]/exit[/]          quit   [{DIM}](Ctrl+C cancels a running question)[/]"""


def fmt_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{round(n / 1000)}k"
    return f"{n / 1e6:.1f}M"


def short_path(p: Path) -> str:
    try:
        return "~/" + p.relative_to(Path.home()).as_posix()
    except ValueError:
        return p.as_posix()


class Session:
    def __init__(self, console: Console, model: str, cwd: Path, auto_yes: bool):
        self.c, self.model, self.cwd, self.auto_yes = console, model, cwd, auto_yes
        self.total_in = self.total_out = self.total_calls = self.questions = self.total_embed = 0
        self.t_in = self.t_out = self.t_calls = 0

    # ---------------------------------------------------------------- model calls (every token is counted here)
    def chat(self, model, messages, tools=None, **kw):
        t0 = time.time()
        with self.c.status(Text("Thinking…", style=ACCENT), spinner="dots", spinner_style=ACCENT):
            r = llm.chat(self.model, messages, tools, **kw)
        self.t_in += r["prompt_tokens"]
        self.t_out += r["completion_tokens"]
        self.t_calls += 1
        self.c.print(Text(f"  ✻ {fmt_tokens(r['prompt_tokens'] + r['completion_tokens'])} tokens · {time.time() - t0:.1f}s",
                          style=DIM))
        return r

    # ---------------------------------------------------------------- progress events from the harness
    def emit(self, kind: str, text: str):
        if kind == "tier":
            num, _, name = text.partition(":")
            self.c.print(Text.assemble(("◆ ", ACCENT), (f"tier {num.strip()}", f"bold {ACCENT}"), (f" · {name.strip()}", DIM)))
        elif kind == "program":
            lines = text.strip().splitlines() or ["(empty)"]
            body = "\n".join(lines[:14]) + (f"\n# … {len(lines) - 14} more lines" if len(lines) > 14 else "")
            self.c.print(Text.assemble(("● ", f"bold {ACCENT}"), ("Python", "bold"), (f"({len(lines)} lines)", DIM)))
            grid = Table.grid(padding=0)
            grid.add_column(width=5)
            grid.add_column()
            grid.add_row(Text("  ⎿  ", style=DIM), Syntax(body, "python", theme="monokai", word_wrap=True,
                                                         background_color=CODE_BG))
            self.c.print(grid)
        elif kind == "excerpts":
            head, _, rest = text.partition("\n")
            self.c.print(Text.assemble(("● ", f"bold {ACCENT}"), ("Retrieve", "bold"), (f"({head})", DIM)))
            lines = rest.splitlines()
            out = Text()
            for i, ln in enumerate(lines[:6]):
                out.append(("  ⎿  " if i == 0 else "     ") + ln[:150] + ("\n" if i < min(len(lines), 6) - 1 else ""), style=DIM)
            self.c.print(out)
        elif kind == "note":
            self.c.print(Text(f"  ! {text}", style="#FFB86C"))
        elif kind == "call":
            name, _, args = text.partition(" ")
            self.c.print(Text.assemble(("● ", f"bold {ACCENT}"), (name, "bold"), (f"({lean._oneline(args, 140)})", DIM)))
        elif kind == "output":
            lines = (text or "(no output)").splitlines() or ["(no output)"]
            out = Text()
            for i, ln in enumerate(lines[:6]):
                out.append(("  ⎿  " if i == 0 else "     ") + ln[:150] + ("\n" if i < min(len(lines), 6) - 1 else ""), style=DIM)
            if len(lines) > 6:
                out.append(f"\n     … +{len(lines) - 6} lines", style=DIM)
            self.c.print(out)

    def approve(self, kind: str, detail: str, risk: str | None = None) -> bool:
        """Ask before running model-written code. A risky action (deletes/writes files, network, installs) is ALWAYS
        asked about, even with --yes or 'always': those only cover ordinary read-only code."""
        if self.auto_yes and not risk:
            return True
        if kind != "python" or risk:  # an ordinary program was already shown
            self.c.print(Text.assemble(("  ? ", "bold #FFB86C"), (f"{kind}: ", "bold"), (lean._oneline(detail, 180), DIM)))
        if risk:
            self.c.print(Text(f"  ⚠ risky: {risk}", style="bold #FFB86C"))
        choices = [("[y]", f"bold {ACCENT}"), ("es  ", DIM)] + ([] if risk else [("[a]", f"bold {ACCENT}"), ("lways  ", DIM)])
        q = Text.assemble(("  Run this ", ""), (kind, "bold"), (f" in {self.cwd.name}/? ", ""), *choices,
                          ("[n]", f"bold {ERR}"), ("o ", DIM))
        try:
            ans = self.c.input(q).strip().lower()
        except EOFError:           # no terminal to ask on (piped input, CI): the safe answer is no
            self.c.print(Text("  no terminal to ask on, so not running it (pass --yes to allow read-only programs)", style=ERR))
            return False
        if ans.startswith("a") and not risk:
            self.auto_yes = True
            return True
        return ans.startswith("y")

    # ---------------------------------------------------------------- one question
    def ask(self, question: str) -> str:
        self.t_in = self.t_out = self.t_calls = 0
        t0 = time.time()
        res = lean.run(question, self.cwd, self.model, chat=self.chat, emit=self.emit, approve=self.approve)
        secs = time.time() - t0
        self.total_in += self.t_in
        self.total_out += self.t_out
        self.total_calls += self.t_calls
        self.questions += 1
        answer = res["final"] or "(no answer)"
        self.c.print()
        self.c.print(Text.assemble(("● ", "bold white"), answer.replace("\n", "\n  ")))   # keep multi-line answers under the bullet
        cost = self.t_in * IN_PRICE + self.t_out * OUT_PRICE
        emb = res.get("embed_tokens", 0)
        self.total_embed += emb
        extra = f" (+{emb:,} embed)" if emb else ""
        self.c.print(Text(f"  ⎿  {self.t_in + self.t_out:,} tokens{extra} (↑{fmt_tokens(self.t_in)} ↓{fmt_tokens(self.t_out)}) · "
                          f"{self.t_calls} call{'s' if self.t_calls != 1 else ''} · tier {res['tier']} · {secs:.1f}s · "
                          f"≈${cost:.4f}", style=DIM))
        self.c.print()
        return answer

    def status(self) -> str:
        cost = self.total_in * IN_PRICE + self.total_out * OUT_PRICE
        return f"{self.model} · ↑{fmt_tokens(self.total_in)} ↓{fmt_tokens(self.total_out)} · ${cost:.4f}"

    def tokens(self):
        total = self.total_in + self.total_out
        cost = self.total_in * IN_PRICE + self.total_out * OUT_PRICE
        self.c.print(Text(f"  session: {self.questions} question{'s' if self.questions != 1 else ''} · "
                          f"{self.total_calls} call{'s' if self.total_calls != 1 else ''} · {total:,} tokens "
                          f"(↑{self.total_in:,} ↓{self.total_out:,}) · ≈${cost:.4f} at ref price", style=DIM))


# -------------------------------------------------------------------- screen pieces

def welcome(s: Session):
    cwd = short_path(s.cwd)
    cwd = cwd if len(cwd) <= 34 else "…" + cwd[-33:]
    info = Text.assemble(("✻ ", ACCENT), ("Welcome to ", "bold"), ("leanloop", f"bold {ACCENT}"), ("!\n\n", "bold"),
                         ("/help for help\n", DIM), (f"cwd: {cwd}", DIM), no_wrap=True)
    grid = Table.grid(padding=(0, 2))
    grid.add_row(Text(LOGO, style=f"bold {ACCENT}", no_wrap=True), info)
    s.c.print(Panel(grid, box=box.ROUNDED, border_style=ACCENT, padding=(0, 1), expand=False))
    s.c.print(Text(f"  model: {s.model} · you approve code before it runs · Ctrl+C cancels", style=DIM))
    s.c.print()


def input_box_top(width: int) -> str:
    return "╭" + "─" * (width - 2) + "╮"


def input_box_bottom(width: int) -> str:
    return "╰" + "─" * (width - 2) + "╯"


class LineReader:
    """Claude Code-style input: a rounded box with a > prompt; below it '? for shortcuts' and a token meter."""

    def __init__(self, s: Session):
        self.s, self.pt = s, None
        if sys.stdin.isatty() and sys.stdout.isatty():
            try:
                from prompt_toolkit import PromptSession
                from prompt_toolkit.history import InMemoryHistory
                from prompt_toolkit.styles import Style
                self.pt = PromptSession(history=InMemoryHistory())
                self.style = Style.from_dict(PT_STYLE)
            except Exception:  # e.g. mintty without a Windows console: use the plain reader
                self.pt = None

    def _width(self) -> int:
        return max(30, shutil.get_terminal_size((80, 24)).columns)   # full width: the right edge lines up with │

    def read(self) -> str:
        w = self._width()
        if self.pt is None:  # plain mode
            self.s.c.print(Text(input_box_top(w), style=BORDER))
            line = self.s.c.input(Text.assemble(("│ ", BORDER), ("> ", f"bold {ACCENT}")))
            self.s.c.print(Text(input_box_bottom(w), style=BORDER))
            return line
        from prompt_toolkit.formatted_text import FormattedText

        def toolbar():
            right = self.s.status()
            left = "  ? for shortcuts"
            gap = max(2, w - len(left) - len(right) - 1)
            return FormattedText([("class:border", input_box_bottom(w) + "\n"),
                                  ("class:bottom-toolbar", left + " " * gap + right)])

        self.s.c.print(Text(input_box_top(w), style=BORDER))      # top edge; the prompt below is a single line
        line = self.pt.prompt(
            FormattedText([("class:border", "│ "), ("class:arrow", "> ")]),
            style=self.style,
            rprompt=FormattedText([("class:border", "│")]),
            bottom_toolbar=toolbar,
            placeholder=FormattedText([("class:placeholder", 'Try "how many ERROR lines are in app.log?"')]),
        )
        self.s.c.print(Text(input_box_bottom(w), style=BORDER))   # close the box in the scrollback
        return line


def repl(s: Session):
    welcome(s)
    reader = LineReader(s)
    while True:
        try:
            line = reader.read().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "?":
            line = "/help"
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                s.c.print(Padding(HELP, (0, 0, 1, 2)))
            elif cmd == "/tokens":
                s.tokens()
            elif cmd == "/model":
                want = arg.strip()
                try:
                    have = llm.list_models()
                except llm.LLMError as e:
                    s.c.print(Text(f"  ✕ {e}", style=ERR))
                    continue
                if not want:
                    s.c.print(Text(f"  model: {s.model}   available: {', '.join(have) or 'none'}", style=DIM))
                elif want in have or f"{want}:latest" in have:
                    s.model = want
                    s.c.print(Text(f"  model → {s.model}", style=DIM))
                else:
                    s.c.print(Text(f"  ✕ no model '{want}'. available: {', '.join(have) or 'none'}", style=ERR))
            elif cmd == "/yes":
                s.auto_yes = not s.auto_yes
                s.c.print(Text(f"  run code without asking: {'on' if s.auto_yes else 'off'}", style=DIM))
            elif cmd == "/clear":
                s.total_in = s.total_out = s.total_calls = s.questions = 0
                s.c.print(Text("  session totals reset", style=DIM))
            else:
                s.c.print(Text(f"  unknown command {cmd} · try /help", style=ERR))
            continue
        try:
            s.ask(line)
        except KeyboardInterrupt:
            s.c.print(Text("  ✕ cancelled", style=ERR))
        except Exception as e:  # keep the session alive (e.g. Ollama not running)
            s.c.print(Text(f"  ✕ error: {e}", style=ERR))
    s.c.print()
    s.tokens()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="leanloop", description="token-lean coding agent (Ollama)")
    ap.add_argument("-p", "--print", dest="question", help="ask one question, print the answer, exit")
    ap.add_argument("--model", default="qwen3-8b-8k")
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--yes", action="store_true", help="run model-written code without asking")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, which can't print ∞ ● ⎿
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    console = Console()
    cwd = Path(a.cwd).resolve()
    if not cwd.is_dir():
        console.print(Text(f"✕ --cwd is not a folder: {cwd}", style=ERR))
        return 2
    try:   # fail early, with instructions, instead of deep inside the first question
        have = llm.list_models()
    except llm.LLMError as e:
        console.print(Text(f"✕ {e}", style=ERR))
        return 2
    if a.model not in have and f"{a.model}:latest" not in have:
        console.print(Text(f"✕ model '{a.model}' is not in Ollama. available: {', '.join(have) or 'none'}", style=ERR))
        return 2
    if not any(m.startswith(prune.EMBED_MODEL) for m in have):
        console.print(Text(f"! {prune.EMBED_MODEL} is not installed, so document questions fall back to the program path "
                           f"(`ollama pull {prune.EMBED_MODEL}`)", style="#FFB86C"))
    s = Session(console, a.model, cwd, a.yes)
    if a.question:
        try:
            answer = s.ask(a.question)
        except KeyboardInterrupt:
            console.print(Text("✕ cancelled", style=ERR))
            return 130
        except Exception as e:
            console.print(Text(f"✕ error: {e}", style=ERR))
            return 1
        return 0 if answer and not answer.startswith(("(no answer)", "Cancelled", "Blocked", "I could not finish")) else 1
    repl(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
