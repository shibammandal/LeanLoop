---
license: mit
task_categories:
- text-generation
tags:
- agent
- trajectory
- agenttune
- auditkit
- lexsi-hackathon
- efficiency
size_categories:
- 1K<n<10K
---

# LeanLoop: Trajectories, Traces, and Benchmark Logs

This repository contains the full evaluation traces, AgentTune-formatted agentic trajectories, and benchmark outputs for **LeanLoop** (submitted to the **Lexsi AI Safety & Alignment Hackathon · Track 3: Efficient Trajectories**).

## Directory Structure

```text
.
├── agenttune/
│   ├── dabench_qwen3_leanloop/
│   │   ├── trajectories.jsonl        # 252 KB (LeanLoop AgentTune trajectory schema)
│   │   └── lexsi_provenance.json
│   └── dabench_qwen3_pi/
│       ├── trajectories.jsonl        # 458 KB (Pi baseline trajectory schema)
│       └── lexsi_provenance.json
│
├── dabench/                          # External DABench evaluation logs (40 tasks)
│   ├── dabench_named_qwen3_lean.jsonl
│   ├── dabench_named_coder_lean.jsonl
│   ├── traj_dabench_qwen3_pi.jsonl
│   └── auditkit_dabench.json         # AuditKIT scoring and step budgets
│
├── bipia/                            # Indirect prompt injection evaluation (80 docs)
│   ├── bipia_qwen3_lean_attack.jsonl
│   ├── bipia_qwen3_lean_clean.jsonl
│   ├── bipia_fresh_lean_new_attack.jsonl
│   ├── bipia_fresh_pi_attack.jsonl
│   ├── bipia_qwen3_lean_scored.json
│   └── bipia_qwen3_pi_scored.json
│
├── held_out_test/                    # In-house held-out benchmark (12 tasks)
│   ├── test_lean.jsonl
│   ├── test_pi.jsonl
│   └── test_naive.jsonl
│
└── safetune/                         # SafeTune activation steering experiment
    └── safetune_guardrail.json
```

## Reproduction & Independent Verification
The AgentTune trajectories can be independently loaded and evaluated using Lexsi's AuditKIT:

```bash
pip install auditkit
python research/lexsi_auditkit.py data/dabench/tasks_named.json auditkit_results.json \
    "leanloop=agenttune/dabench_qwen3_leanloop" "pi=agenttune/dabench_qwen3_pi"
```
