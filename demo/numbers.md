## Held-out test (12 tasks): tokens, correctness, model calls

| harness | correct | model calls | calls/task | total tokens | tokens/task | tokens per correct answer | $ ref |
|---|---|---|---|---|---|---|---|
| pi (unmodified) | 7/12 | 35 | 2.9 | 181,459 | 15,121 | 25,922 | $0.747 |
| naive loop | 7/12 | 28 | 2.3 | 24,828 | 2,069 | 3,546 | $0.084 |
| leanloop v0 (paging only) | 5/12 | 36 | 3.0 | 19,020 | 1,585 | 3,804 | $0.070 |
| **leanloop** (5 runs: best / worst) | 9/12 (all 5 runs) | 14–18 | 1.2–1.5 | 5,859–8,649 | 488–720 | 651–961 | $0.030–0.040 |

all leanloop runs: correct=[9, 9, 9, 9, 9] tokens=[5911, 7324, 5859, 8647, 8649] calls=[14, 16, 14, 18, 18]

vs pi: worst run saves 95.2% of tokens, best 96.8%; calls 18–14 vs 35
tokens per correct answer: pi 25,922 vs leanloop 651–961 (27x–40x fewer)
vs naive loop (same file Q&A tools, full history): worst run saves 65%, best 76%

## Documents + prompt injection (24 runs per mode per bench run; two bench runs)

| context sent | run | correct | generic injection won | topical poisoning won | model tokens/question |
|---|---|---|---|---|---|
| all documents | 1 | 21/24 | 1/24 | 2/24 | 2,891 |
| all documents | 2 | 20/24 | 2/24 | 2/24 | 2,891 |
| top-2 whole docs | 1 | 20/24 | 0/24 | 3/24 | 620 |
| top-2 whole docs | 2 | 19/24 | 0/24 | 4/24 | 620 |
| micro-spans | 1 | 21/24 | 0/24 | 3/24 | 189 |
| micro-spans | 2 | 21/24 | 0/24 | 3/24 | 189 |
| micro-spans + guard | 1 | 21/24 | 0/24 | 3/24 | 189 |
| micro-spans + guard | 2 | 21/24 | 0/24 | 3/24 | 189 |
