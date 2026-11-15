"""AgentDojo feasibility probe: can a small LOCAL model (through Ollama) run AgentDojo at all, and what does it cost?

This measures the MODEL + AgentDojo's own tool-calling pipeline. It is NOT leanloop: leanloop's program-first cascade has no
adapter for AgentDojo's function-calling tool environments (that would be a separate project). Token counts come from wrapping
the OpenAI client that AgentDojo's LocalLLM uses.

Run it with the venv that has agentdojo installed:
  <venv>/python research/agentdojo_probe.py <suite> <n_user_tasks> <n_injection_tasks> [model]
n_injection_tasks=0 -> utility only (no attack).
"""
import os
import sys
import time

os.environ.setdefault("LOCAL_LLM_PORT", "11434")      # Ollama's OpenAI-compatible endpoint
import openai.resources.chat.completions as occ  # noqa: E402

TOK = {"prompt": 0, "completion": 0, "calls": 0}
_orig = occ.Completions.create


def _counting_create(self, *a, **k):
    r = _orig(self, *a, **k)
    u = getattr(r, "usage", None)
    TOK["calls"] += 1
    if u:
        TOK["prompt"] += u.prompt_tokens
        TOK["completion"] += u.completion_tokens
    return r


occ.Completions.create = _counting_create

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.logging import OutputLogger  # noqa: E402
import tempfile  # noqa: E402
from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections  # noqa: E402
from agentdojo.task_suite import get_suite  # noqa: E402

VERSION = "v1.2.1"


def main(suite_name, n_user, n_inj, model="qwen3-8b-8k"):
    n_user, n_inj = int(n_user), int(n_inj)
    suite = get_suite(VERSION, suite_name)
    from agentdojo.agent_pipeline.agent_pipeline import load_system_message
    sysmsg = load_system_message(None)
    if model.startswith("qwen3"):          # Qwen3 thinks by default through Ollama's OpenAI endpoint; its soft switch turns that off
        sysmsg = "/no_think\n" + sysmsg
    pipeline = AgentPipeline.from_config(PipelineConfig(llm="local", model_id=model, defense=None, system_message_name=None,
                                                        system_message=sysmsg))
    pipeline.name = f"local-{model}"
    users = [f"user_task_{i}" for i in range(n_user)]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as logdir, OutputLogger(logdir):     # AgentDojo needs an active logger
        run(suite, suite_name, pipeline, users, n_user, n_inj)
    print(f"model calls {TOK['calls']}, prompt tokens {TOK['prompt']:,}, completion tokens {TOK['completion']:,}, "
          f"total {TOK['prompt'] + TOK['completion']:,}, seconds {time.time() - t0:.0f}")


def run(suite, suite_name, pipeline, users, n_user, n_inj):
    if n_inj == 0:
        res = benchmark_suite_without_injections(pipeline, suite, logdir=None, force_rerun=True, user_tasks=users, benchmark_version=VERSION)
        util = list(res["utility_results"].values())
        print(f"{suite_name}: utility {sum(util)}/{len(util)} on {n_user} user tasks (no attack)")
    else:
        attack = load_attack("important_instructions", suite, pipeline)
        inj = [f"injection_task_{i}" for i in range(n_inj)]
        res = benchmark_suite_with_injections(pipeline, suite, attack, logdir=None, force_rerun=True, user_tasks=users,
                                              injection_tasks=inj, verbose=False, benchmark_version=VERSION)
        util, sec = list(res["utility_results"].values()), list(res["security_results"].values())
        print(f"{suite_name}: under attack: utility {sum(util)}/{len(util)}, attack success {sum(sec)}/{len(sec)} "
              f"({n_user} user tasks x {n_inj} injection tasks)")


if __name__ == "__main__":
    main(*sys.argv[1:])
