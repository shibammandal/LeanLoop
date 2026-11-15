
## DABench, 40 questions (seed 2026, balanced easy/medium/hard): file named in the prompt (benchmark protocol)

| model | harness | correct | lenient | easy | medium | hard | calls | calls/q | total tokens | tokens/q | tokens per correct | answered by tier 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 8B | lean | 24/40 | 26/40 | 11/14 | 8/13 | 5/13 | 97 | 2.4 | 89,321 | 2,233 | 3,721 | 33/40 |
| Qwen3 8B | naive | 15/40 | 18/40 | 8/14 | 5/13 | 2/13 | 296 | 7.4 | 1,060,944 | 26,523 | 70,729 | - |
| Qwen3 8B | pi | 14/40 | 16/40 | 5/14 | 6/13 | 3/13 | 157 | 3.9 | 1,044,546 | 26,113 | 74,610 | - |
| Qwen3 8B | pi (run 2, traced) | 12/40 | 12/40 | 5/14 | 5/13 | 2/13 | 141 | 3.5 | 909,131 | 22,728 | 75,760 | - |
| Qwen2.5-Coder 7B | lean | 25/40 | 27/40 | 11/14 | 9/13 | 5/13 | 79 | 2.0 | 51,546 | 1,288 | 2,061 | 27/40 |

Paired comparison on the same questions (exact sign test; small n):

- Qwen3 8B: leanloop vs naive: leanloop-only correct 12, naive-only correct 3 (p=0.04); tokens +92% (89,321 vs 1,060,944); calls 97 vs 296
- Qwen3 8B: leanloop vs pi: leanloop-only correct 15, pi-only correct 5 (p=0.04); tokens +91% (89,321 vs 1,044,546); calls 97 vs 157
- Qwen3 8B: leanloop vs pi (run 2, traced): leanloop-only correct 15, pi (run 2, traced)-only correct 3 (p=0.01); tokens +90% (89,321 vs 909,131); calls 97 vs 141
- Qwen3 8B leanloop: 7/40 questions needed the fallback loop and used 68,871 of its 89,321 tokens; 0 had a program blocked by the read-only check
- Qwen2.5-Coder 7B leanloop: 13/40 questions needed the fallback loop and used 33,917 of its 51,546 tokens; 0 had a program blocked by the read-only check

## DABench, 40 questions (seed 2026, balanced easy/medium/hard): file NOT named (harness must find it)

| model | harness | correct | lenient | easy | medium | hard | calls | calls/q | total tokens | tokens/q | tokens per correct | answered by tier 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3 8B | lean | 26/40 | 26/40 | 12/14 | 9/13 | 5/13 | 111 | 2.8 | 102,200 | 2,555 | 3,930 | 30/40 |
| Qwen3 8B | naive | 4/40 | 5/40 | 2/14 | 2/13 | 0/13 | 400 | 10.0 | 1,176,300 | 29,407 | 294,075 | - |
| Qwen3 8B | pi | 5/40 | 6/40 | 2/14 | 2/13 | 1/13 | 106 | 2.6 | 640,536 | 16,013 | 128,107 | - |
| Qwen2.5-Coder 7B | lean | 24/40 | 25/40 | 9/14 | 10/13 | 5/13 | 79 | 2.0 | 49,000 | 1,225 | 2,041 | 28/40 |

Paired comparison on the same questions (exact sign test; small n):

- Qwen3 8B: leanloop vs naive: leanloop-only correct 24, naive-only correct 2 (p=0.00); tokens +91% (102,200 vs 1,176,300); calls 111 vs 400
- Qwen3 8B: leanloop vs pi: leanloop-only correct 21, pi-only correct 0 (p=0.00); tokens +84% (102,200 vs 640,536); calls 111 vs 106
- Qwen3 8B leanloop: 10/40 questions needed the fallback loop and used 85,442 of its 102,200 tokens; 0 had a program blocked by the read-only check
- Qwen2.5-Coder 7B leanloop: 12/40 questions needed the fallback loop and used 30,517 of its 49,000 tokens; 0 had a program blocked by the read-only check

## BIPIA indirect prompt injection (Microsoft): 40 documents (20 email, 20 table), one planted instruction each

| harness | model | planted instruction followed | still answered the question | same questions, no attack | tokens | calls | judge catches real completions |
|---|---|---|---|---|---|---|---|
| leanloop | Qwen3 8B | 0/40 | 16/40 | 18/40 | 10,939 | 48 | 30/40 |
| leanloop | Qwen2.5-Coder 7B | 0/40 | 15/40 | 17/40 | 9,724 | 46 | 31/40 |
| pi (unmodified) | Qwen3 8B | 13/40 | 21/40 | 28/40 | 406,260 | 86 | 31/40 |
| whole document + leanloop's strict prompt (ablation) | Qwen3 8B | 3/40 | 18/40 | - | 16,659 | 40 | - |

- followed the planted instruction: pi 13/40 vs leanloop 0/40 (Fisher exact p=0.00008)
- pi 13/40 vs whole document + strict prompt 3/40 (p=0.010); whole document 3/40 vs leanloop's pruned 0/40 (p=0.24): pruning's extra effect is not significant
- tokens: leanloop 10,939 vs pi 406,260 (97% fewer)
- clean-copy accuracy on email questions: pi 12/20 vs leanloop 10/20
- clean-copy accuracy on table questions: pi 16/20 vs leanloop 8/20
- what true attack rate is still plausible after seeing 0/40 from a judge that catches 75% of real completions: 5% -> P(0/40)=22%; 10% -> P(0/40)=4%; 15% -> P(0/40)=1%
- did the planted text reach leanloop's model? email: 12/20 (removed by guard 1); table: 1/20 (removed by guard 0)

## Table-routing fix, validated on 40 FRESH BIPIA documents (none used before; Qwen3 8B)

| | planted instruction followed | still answered (attacked) | same documents, no attack | tokens | calls |
|---|---|---|---|---|---|
| leanloop, table routing OFF (before) | 1/40 | 17/40 | 18/40 | 24,440 | 62 |
| leanloop, table routing ON (after) | 4/40 | 13/40 | 16/40 | 73,591 | 122 |
| pi (unmodified) | 14/40 | 19/40 | 23/40 | 419,075 | 88 |
- clean-copy accuracy on table questions (of 20): leanloop table routing OFF (before): 8; leanloop table routing ON (after): 6; pi (unmodified): 11
- clean-copy accuracy on email questions (of 20): leanloop table routing OFF (before): 10; leanloop table routing ON (after): 10; pi (unmodified): 12
- table accuracy after vs before: 6/20 vs 8/20 (Fisher exact p=0.741)
- planted instruction followed: after 4/40, before 1/40, pi 14/40
- fresh set, pi 14/40 vs leanloop (as validated) 1/40: Fisher exact p=0.0003
- both sets pooled (80 documents): planted instruction followed pi 27/80 (34%) vs leanloop 1/80 (1%), p=1.8e-08
- both sets pooled: accuracy on the clean copies pi 51/80 (64%) vs leanloop 36/80 (45%), p=0.026
- tokens: leanloop uses 94% fewer than pi on the fresh set

## Lexsi AuditKIT cross-check: DABench trajectories exported in AgentTune's format, scored by AuditKIT's agent evaluation

Verdicts come from AuditKIT's runner (the DABench rule as a versioned `CustomPredicate`); tokens come from our recorded counts, placed in AuditKIT's own `resource_use.tokens` slot. A step is one model call.

| harness | AuditKIT success | agrees with our scorer | within 1 call | within 2 calls | within 3 calls | within 4 calls | within 6 calls | within 10 calls | tokens | tokens per success |
|---|---|---|---|---|---|---|---|---|---|---|
| leanloop (Qwen3 8B) | 24/40 | 40/40 | 18 | 22 | 22 | 24 | 24 | 24 | 89,321 | 3,722 |
| pi (Qwen3 8B, run 2) | 12/40 | 40/40 | 0 | 8 | 11 | 12 | 12 | 12 | 909,131 | 75,761 |

## Lexsi SafeTune: can refusal-direction steering restore refusals that an attack removed? (separate experiment; Qwen/Qwen2.5-0.5B-Instruct on CPU; 30 harmful + 30 benign test prompts; attack = prefill 'Sure, here is')

Validation sweep (steering settings tried; the best by 'refuses attacked-harmful minus refuses benign' was chosen, never using the test prompts):

| setting | refuses attacked-harmful | refuses benign (over-refusal) | score |
|---|---|---|---|
| all layers, strength 0.2 | 0% | 50% | -0.50 |
| all layers, strength 0.3 | 42% | 83% | -0.42 |
| layer 18 only, strength 8.0 | 0% | 50% | -0.50 |
| layer 18 only, strength 16.0 **(chosen)** | 67% | 92% | -0.25 |
| layer 18 only, strength 32.0 | 67% | 92% | -0.25 |

Test prompts:

| condition | refuses harmful requests | refuses benign requests (over-refusal) | extra prompt tokens per request |
|---|---|---|---|
| no attack, no safeguard (the model's normal behaviour) | 100% | 3% | 0 |
| attack, no safeguard | 0% | 3% | 0 |
| attack + safety system prompt | 0% | 0% | 38.0 |
| attack + SafeTune steering (layer 18 only, strength 16.0) | 43% | 83% | 0.0 |

Empty outputs: 0 of 360. Runtime 48 minutes.

## AgentDojo feasibility probe (ETH Zurich): banking suite, Qwen3 8B through AgentDojo's own tool-calling pipeline

This is the model + AgentDojo's pipeline, NOT leanloop (no adapter exists for its tool environments).

- no attack: banking: utility 5/8 on 8 user tasks (no attack) | model calls 22, prompt tokens 52,953, completion tokens 15,739, total 68,692, seconds 424
- important_instructions attack: banking: under attack: utility 2/8, attack success 3/8 (4 user tasks x 2 injection tasks) | model calls 29, prompt tokens 72,729, completion tokens 28,044, total 100,773, seconds 748
