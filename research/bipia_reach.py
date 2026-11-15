"""Mechanism check for BIPIA: how often does the planted instruction actually REACH the model in leanloop's document tier?

Replays exactly what tier R does (split the attacked file into spans, rank them against the prompt with the local embedder, keep
the top few) and checks whether the attack text survives into the kept spans, or is removed by the injection guard.
Needs Ollama running with nomic-embed-text. Reads untrusted data, so run it with `python -I`.

usage: python -I research/bipia_reach.py <suite_dir> [out.json]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from leanloop import lean, prune  # noqa: E402


def main(suite, out=None):
    suite = Path(suite)
    tasks = json.loads((suite / "tasks.json").read_text(encoding="utf-8"))
    n, hit = Counter(), Counter()
    for t in tasks:
        spans = prune.split_spans(lean._read_text(suite / "ws" / t["id"] / t["file"]))
        r = prune.prune(t["prompt"], spans, embed=lean._EMBED)          # the same call tier R makes
        probe = t["attack"][:40]
        n[t["source"]] += 1
        hit[(t["source"], "attack_text_in_document")] += any(probe in s for s in spans)
        hit[(t["source"], "reached_the_model")] += any(probe in s for s in r.spans)
        hit[(t["source"], "removed_by_guard")] += any(probe in s for s in r.dropped)
    res = {src: {"n": n[src], **{k: hit[(src, k)] for k in ("attack_text_in_document", "reached_the_model", "removed_by_guard")}}
           for src in n}
    for src, v in res.items():
        print(f"{src}: attack in document {v['attack_text_in_document']}/{v['n']} | reached the model {v['reached_the_model']}/{v['n']} "
              f"| removed by the guard {v['removed_by_guard']}/{v['n']}")
    if out:
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
