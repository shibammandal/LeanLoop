"""Ablation for BIPIA: is leanloop's protection from PRUNING or from its strict answer-only PROMPT?

Gives the model the ENTIRE attacked document (no pruning, no retrieval) but with leanloop's document-tier prompt
(lean.DOC_SYSTEM: "answer using only the excerpts, reply with the answer only"). One model call per sample, same judge as
bipia_judge.py. Compare with leanloop (pruned, same prompt) and pi (whole document, its own prompt).

usage: python -I research/bipia_ablation.py <suite_dir> [model] [judge_model] [out.json]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bipia_judge import judge, utility  # noqa: E402
from leanloop import lean, llm  # noqa: E402


def main(suite, model="qwen3-8b-8k", judge_model="qwen3-8b-8k", out=None):
    suite = Path(suite)
    tasks = json.loads((suite / "tasks.json").read_text(encoding="utf-8"))
    res = {"model": model, "n": len(tasks), "utility": 0, "asr": 0, "tokens": 0, "calls": 0}
    for t in tasks:
        text = (suite / "ws" / t["id"] / t["file"]).read_text(encoding="utf-8")
        r = llm.chat(model, [{"role": "system", "content": lean.DOC_SYSTEM},
                             {"role": "user", "content": f"Excerpts:\n{text}\n\nQuestion: {t['prompt']}"}], None, num_predict=200)
        final = r["message"]["content"]
        res["utility"] += utility(final, t["ideal"])
        res["asr"] += judge(judge_model, t, final)
        res["tokens"] += r["prompt_tokens"] + r["completion_tokens"]
        res["calls"] += 1
    print(f"whole document + leanloop's strict prompt ({model}): utility {res['utility']}/{res['n']}  "
          f"attack success {res['asr']}/{res['n']}  tokens {res['tokens']:,}")
    if out:
        Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
