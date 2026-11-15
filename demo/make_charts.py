"""Draw the README / write-up charts as standalone SVGs from demo/summarize.py (no hand-typed numbers).

Form: horizontal bars (magnitude comparison, one measure, one hue, so no legend). Spec follows the data-viz skill:
bars <= 24 px with a 4 px rounded data end and a square baseline end, hairline gridlines, values at the bar tips,
text in ink tokens (never the series colour), light and dark surfaces, a <title> tooltip per bar, role=img + desc.
usage: python demo/make_charts.py
"""
import json
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize  # noqa: E402

OUT = Path(__file__).resolve().parent

CSS = """
.s{fill:#fcfcfb;stroke:#e2e8f0;stroke-width:1}.t{fill:#0f172a}.u{fill:#64748b}.m{fill:#94a3b8}.g{stroke:#e2e8f0;stroke-dasharray:2,2}.a{stroke:#475569;stroke-width:2}
.b{fill:#64748b}
.p{fill:#7c3aed}
@media (prefers-color-scheme: dark){
  .s{fill:#0f0e17;stroke:#2d2938}.t{fill:#f8fafc}.u{fill:#94a3b8}.g{stroke:#262330;stroke-dasharray:2,2}.a{stroke:#64748b;stroke-width:2}
  .b{fill:#475569}
  .p{fill:#a78bfa}
}
text{font-family:'Segoe UI Mono','Cascadia Code','Courier New',monospace}
"""


def compact(v):
    return f"{v / 1000:g}K" if v >= 1000 else f"{v:g}"


def bar_path(x, y, w, h, p=4):
    """Pixel-art style bar: crispEdges, 4px stepped notched corner on data end."""
    w = max(int(round(w)), 3)
    h = int(round(h))
    if w <= p * 2:
        return f"M{x},{y} H{x + w} V{y + h} H{x} Z"
    return f"M{x},{y} H{x + w - p} V{y + p} H{x + w} V{y + h - p} H{x + w - p} V{y + h} H{x} Z"


def chart(path, title, subtitle, rows, ticks, desc):
    """rows: (label, sublabel, value, value_text, tooltip)."""
    W, LEFT, AREA, TOP, ROW, H = 780, 250, 380, 78, 58, 0
    H = TOP + ROW * len(rows) + 46
    vmax = ticks[-1]
    sx = lambda v: LEFT + AREA * v / vmax
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'shape-rendering="crispEdges" aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>',
         f"<style>{CSS}</style>", f'<rect class="s" width="{W}" height="{H}" rx="0"/>',
         f'<text class="t" x="24" y="32" font-size="15" font-weight="700">{escape(title)}</text>',
         f'<text class="u" x="24" y="52" font-size="11">{escape(subtitle)}</text>']
    for tv in ticks:
        x = sx(tv)
        o.append(f'<line class="{"a" if tv == 0 else "g"}" x1="{x}" y1="{TOP - 8}" x2="{x}" y2="{H - 34}" stroke-width="1"/>')
        o.append(f'<text class="m" x="{x}" y="{H - 14}" font-size="11" text-anchor="middle">{compact(tv)}</text>')
    for i, (label, sub, v, vtext, tip) in enumerate(rows):
        y = TOP + ROW * i + 10
        is_lean = "leanloop" in label.lower() or "micro-spans" in label.lower()
        bar_cls = "p" if is_lean else "b"
        txt_cls = "p" if is_lean else "t"
        w = max(sx(v) - LEFT, 3)
        o.append(f'<g><title>{escape(tip)}</title>')
        o.append(f'<text class="{"p" if is_lean else "t"}" x="24" y="{y + 12}" font-size="13" font-weight="{700 if is_lean else 400}">{escape(label)}</text>')
        o.append(f'<text class="u" x="24" y="{y + 29}" font-size="11">{escape(sub)}</text>')
        o.append(f'<path class="{bar_cls}" d="{bar_path(LEFT, y, w, 22)}"/>')
        o.append(f'<text class="{txt_cls}" x="{sx(v) + 8}" y="{y + 16}" font-size="12" font-weight="{700 if is_lean else 400}">{escape(vtext)}</text></g>')
    o.append("</svg>")
    Path(path).write_text("\n".join(o), encoding="utf-8")


def main():
    n = summarize.numbers()
    b, runs, lo, hi = n["base"], n["lean_runs"], n["lean_best"], n["lean_worst"]
    fmt = lambda t: f"{t['ok']}/{t['n']} correct"
    lean_sub = f"{lo['ok']}/{lo['n']} correct in all {len(runs)} runs"

    chart(OUT / "chart_tokens.svg", "Tokens to answer 12 held-out questions",
          "Lower is better. Same model for every row: Qwen3 8B via Ollama, 8k context. Bar = worst of 5 leanloop runs.",
          [("pi (unmodified)", fmt(b["pi"]), b["pi"]["tokens"], f"{b['pi']['tokens']:,}",
            f"pi: {b['pi']['tokens']:,} tokens, {fmt(b['pi'])}"),
           ("naive loop", fmt(b["naive"]) + " · full history re-sent", b["naive"]["tokens"], f"{b['naive']['tokens']:,}",
            f"naive loop: {b['naive']['tokens']:,} tokens, {fmt(b['naive'])}"),
           ("leanloop", lean_sub, hi["tokens"], f"{lo['tokens']:,} – {hi['tokens']:,}",
            f"leanloop: {lo['tokens']:,} to {hi['tokens']:,} tokens across {len(runs)} runs, {lean_sub}")],
          [0, 50000, 100000, 150000, 200000],
          f"Bars of total tokens used by pi ({b['pi']['tokens']:,}), a naive loop ({b['naive']['tokens']:,}) and leanloop "
          f"({lo['tokens']:,} to {hi['tokens']:,} across {len(runs)} runs) on 12 held-out questions.")

    chart(OUT / "chart_calls.svg", "Model calls to answer 12 held-out questions",
          "Lower is better. A call is one request to the model; fewer calls means fewer steps and less redundant tool use.",
          [("pi (unmodified)", fmt(b["pi"]), b["pi"]["calls"], f"{b['pi']['calls']}  ({b['pi']['calls'] / 12:.1f} per question)",
            f"pi: {b['pi']['calls']} calls"),
           ("naive loop", fmt(b["naive"]), b["naive"]["calls"], f"{b['naive']['calls']}  ({b['naive']['calls'] / 12:.1f} per question)",
            f"naive loop: {b['naive']['calls']} calls"),
           ("leanloop", lean_sub, hi["calls"], f"{lo['calls']} – {hi['calls']}  ({lo['calls'] / 12:.1f}–{hi['calls'] / 12:.1f} per question)",
            f"leanloop: {lo['calls']} to {hi['calls']} calls across {len(runs)} runs")],
          [0, 10, 20, 30, 40],
          f"Bars of model calls: pi {b['pi']['calls']}, naive loop {b['naive']['calls']}, leanloop {lo['calls']} to {hi['calls']}.")

    d = n["docs"]
    avg = lambda m: d[m]["runs"][next(iter(d[m]["runs"]))]["tok"] // d[m]["runs"][next(iter(d[m]["runs"]))]["n"]
    okr = lambda m: sorted(r["ok"] for r in d[m]["runs"].values())
    rng = lambda m: f"{okr(m)[0]}–{okr(m)[-1]}/24" if okr(m)[0] != okr(m)[-1] else f"{okr(m)[0]}/24"
    chart(OUT / "chart_docs.svg", "Tokens the model reads per document question",
          "Lower is better. 8 questions x 3 workspaces (clean, injected, poisoned), 10 documents of ~2,000 words in total.",
          [("All documents sent", f"{rng('full')} correct", avg("full"), f"{avg('full'):,}", f"all documents: {avg('full'):,} tokens per question"),
           ("Top-2 whole documents", f"{rng('rag2')} correct", avg("rag2"), f"{avg('rag2'):,}", f"top-2 documents: {avg('rag2'):,} tokens per question"),
           ("Micro-spans (leanloop)", f"{rng('span')} correct · 0 generic injections won", avg("span"), f"{avg('span'):,}",
            f"micro-spans: {avg('span'):,} tokens per question")],
          [0, 1000, 2000, 3000],
          f"Bars of model tokens per question: all documents {avg('full'):,}, top-2 documents {avg('rag2'):,}, micro-spans {avg('span'):,}.")
    external_charts()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("chart_*.svg"))))


def external_charts():
    """Charts for the external benchmarks (DABench, BIPIA), read straight from the saved result files."""
    R = OUT.parent / "results"
    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]          # noqa: E731
    dn = {h: load(R / f"dabench_named_qwen3_{h}.jsonl") for h in ("pi", "naive", "lean")}
    tok = {h: sum(r["tokens"] for r in rs) for h, rs in dn.items()}
    ok = {h: f"{sum(bool(r['correct']) for r in rs)}/{len(rs)} correct" for h, rs in dn.items()}
    chart(OUT / "chart_dabench.svg", "Tokens to answer 40 DABench questions (written by others)",
          "Lower is better. Qwen3 8B through Ollama; the data file is named in the prompt, as the benchmark specifies.",
          [("pi (unmodified)", ok["pi"], tok["pi"], f"{tok['pi']:,}", f"pi: {tok['pi']:,} tokens, {ok['pi']}"),
           ("naive loop", ok["naive"] + " · full history re-sent", tok["naive"], f"{tok['naive']:,}", f"naive loop: {tok['naive']:,} tokens, {ok['naive']}"),
           ("leanloop", ok["lean"], tok["lean"], f"{tok['lean']:,}", f"leanloop: {tok['lean']:,} tokens, {ok['lean']}")],
          [0, 400000, 800000, 1200000],
          f"Bars of total tokens on 40 DABench questions: pi {tok['pi']:,}, naive loop {tok['naive']:,}, leanloop {tok['lean']:,}.")

    pi, lean_ = json.loads((R / "bipia_qwen3_pi_scored.json").read_text()), json.loads((R / "bipia_qwen3_lean_scored.json").read_text())
    pi2, lean2 = json.loads((R / "bipia_fresh_pi_scored.json").read_text()), json.loads((R / "bipia_fresh_lean_old_scored.json").read_text())
    abl = json.loads((R / "bipia_ablation_fullctx_qwen3.json").read_text())
    row = lambda name, sub, s: (name, sub, s["asr"], f"{s['asr']} / 40", f"{name}: followed {s['asr']} of 40 planted instructions")   # noqa: E731
    chart(OUT / "chart_bipia.svg", "Planted instructions followed, out of 40 (BIPIA)",
          "Lower is better. Qwen3 8B throughout. Two different sets of 40 documents; judge is a local model using BIPIA's prompts.",
          [row("pi (unmodified), first documents", f"answered {pi['utility']}/40 questions", pi),
           row("pi (unmodified), fresh documents", f"answered {pi2['utility']}/40 questions", pi2),
           row("Whole document, strict prompt", f"first documents · answered {abl['utility']}/40", abl),
           row("leanloop, first documents", f"answered {lean_['utility']}/40 · document pruned to 5 sentences", lean_),
           row("leanloop, fresh documents", f"answered {lean2['utility']}/40 · document pruned to 5 sentences", lean2)],
          [0, 10, 20, 30, 40],
          f"Bars of planted instructions followed out of 40: pi {pi['asr']} and {pi2['asr']} on two document sets, whole document with strict prompt "
          f"{abl['asr']}, leanloop {lean_['asr']} and {lean2['asr']}.")


if __name__ == "__main__":
    main()
