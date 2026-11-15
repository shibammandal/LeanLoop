# Demo script (about 3 minutes)

## Before you start (1 minute, do this off-camera)

```
ollama serve                                   # if it is not already running
ollama list                                    # expect qwen3-8b-8k and nomic-embed-text
cd ~/leanloop && python -m pytest -q           # 144 passed
cp -r data/ws2 ~/demo_data && cp -r data/docs/generic ~/demo_docs
```
Use Windows Terminal or PowerShell (the input box and status line need a real terminal). Warm the model once with `hey`
so the first scene is not slow. **Close the browser and other heavy apps first**: an 8B model plus a busy desktop runs out of
memory on a 16 GB laptop.

**If anything misbehaves live:** open `demo/demo.svg` (a recording of the real run) and `demo/demo_transcript.txt`, and
say so. The numbers on screen are the same ones in `SUBMISSION.md`.

## The talk track

**0:00 Problem (20 s).** "Track 3 asks for fewer steps and fewer redundant tool calls without lowering success. Agent
harnesses re-send a huge prefix on every call and need about three calls per question. On the same local model, pi needed 141
to 157 model calls and about a million tokens to answer 40 data-analysis questions, and got 12 to 14 right."

**0:20 Idea (20 s).** "leanloop tries the cheapest path first and checks it with plain rules. Usually the model writes one
short program, we run it, and its output is the answer. One call, no tool schemas. It falls back to a full agent loop only
if the check fails." Show `demo/chart_dabench.svg`.

**0:40 Live: `cd ~/demo_data && leanloop`**

1. `hey` → "Tier 0. Seventy tokens, no files touched."
2. `How many ERROR lines in app.log have code=E102?` → "One call, about 430 tokens. The program is shown, I approve it,
   the output is the answer."
3. `Delete every file in this folder.` → "The model wrote `os.remove`. It was **not run**, even though I'm allowed to run
   code. It says why." Then `ls` in another window: files unchanged. (This is the safety point. Pause here.)
4. `cd ~/demo_docs && leanloop`, then `After how many minutes of inactivity does a VPN session time out?` → "These documents have
   planted 'ignore all previous instructions' lines. Only 5 of 146 sentences reach the model; the guard shows what it dropped.
   About 200 tokens instead of 2,900."
5. `How many rows are in missing_file.csv?` → "No model call at all."

**2:15 Result (45 s).** "On questions we did not write. DABench: 24 of 40 against 12 to 14 for pi, with 90 percent fewer
tokens. Lexsi's own AuditKIT re-scored the runs and agreed on every verdict, and with its step budgets it shows leanloop
solving 22 questions within two model calls where pi solves 8. On the BIPIA injection benchmark, across 80 documents, pi
followed the planted instruction 27 times and leanloop once. Show `demo/chart_bipia.svg`."

**3:00 Honesty (30 s).** "Three things I want you to know. On document questions pi is more accurate, 64 percent against
45. Most of leanloop's injection protection comes from how it frames the question, not the pruning; I tested that. And a
table fix I tried did not work on fresh documents, so it is off. The read-only check is a safety net, not a sandbox."

## Questions you will probably get

- **"Why didn't you train anything?"** Deliberate: the savings come from system design, and we can show them without a
  training run. The trajectories are already exported in AgentTune's format; distilling them into a small model is the
  natural next step.
- **"Which Lexsi tools did you use, and how?"** AuditKIT: it re-scored our runs through its agent-evaluation layer (a
  versioned custom outcome for the DABench rule), and we fill its `resource_use.tokens` slot, which it has but cannot fill
  from AgentTune files. Its per-case step budgets answer Track 3's question. AgentTune: its trajectory format, read back
  through AuditKIT's importer; we did not install AgentTune or use its trainer. SafeTune: a separate experiment on a small
  model. Steering restored 43% of the refusals a prefill attack removed, at zero extra tokens, where a 38-token safety prompt
  restored none, but it made the model refuse 83% of harmless requests, so it is not a usable guardrail as it stands.
- **"Isn't pi just handicapped by its big system prompt?"** Partly, yes (about 3,900 tokens per call). That is why the
  naive-loop row exists: same tools, no big prompt. leanloop still uses 65–76% fewer tokens on our benchmark and 92% fewer on
  DABench, and was more accurate.
- **"Is it really more accurate than pi?"** On DABench, 24 against 12 and 14 in two pi runs, p between 0.01 and 0.04 for
  three overlapping tests, and the leanloop run was repeated once after a bug fix. I would say "more accurate on this
  sample", not "proven". On document questions it is the other way round.
- **"Is 12 questions enough?"** No. It shows the mechanism, not a general claim, which is why we added 40 DABench questions
  and 80 BIPIA documents we did not write. Even those are small, and leanloop was run once per condition.
- **"Why do tokens vary between runs?"** Ollama's GPU sampling is not bit-reproducible at temperature 0. On our own set,
  eight of 12 questions are within 3 tokens every run; 2–3 borderline ones sometimes need the fallback.
- **"Does the guard stop prompt injection?"** The guard stops *generic* injected instructions. The protection that matters in
  BIPIA is mostly prompt framing: the same model given the whole document with the strict prompt followed 3 of 40, against
  pi's 13. It does **not** stop topical poisoning (3 of 24 got through); an experimental conflict warning flags it on our
  own documents but is off by default and was designed on that data.
- **"Why 1 of 80 and not 0?"** The judge is a local model that misses about a quarter of real attacks, so any rate is a lower
  bound. We say so.
- **"Did you tune on those benchmarks?"** No tuning on DABench or BIPIA. We made corrections after seeing results, all
  disclosed: the first DABench runs forgot to name the data file, and a bug in our free "no such file" rule (file names with
  spaces) cost leanloop 4 of 40 questions. We fixed it, re-ran, and kept the old runs. We also tried a table-handling fix,
  tested it on fresh documents, and it made things worse, so it is off.
- **"Why is there no pi number for Qwen2.5-Coder?"** That model prints its tool calls as text, so pi would execute nothing;
  any pi score would measure that quirk, not pi. We report leanloop on it (25 of 40, 51,546 tokens).
- **"Can the read-only check be bypassed?"** Yes, by a determined model: it is static analysis. The real fix is an OS-level
  sandbox. It stops the accidental and the common case, and risky actions always ask.
- **"What about AgentDojo?"** A feasibility probe only: it installs and runs a local model, and the baseline model followed the
  attacker in 3 of 8 runs on its banking suite. That measures the model plus AgentDojo's pipeline, not leanloop, which has no
  adapter for its tool environments yet.
