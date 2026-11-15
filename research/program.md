# leanloop autoresearch

This copies the pattern of karpathy/autoresearch: an agent edits one file, runs a fixed evaluation, keeps the change if the score improves and throws it away if not, then repeats.

## Files
- `research/prepare.py` is **fixed**. It builds the workspace `data/ws2` and two task sets: `dev`, which the loop tunes on, and `test`, which is held back.
- `research/evaluate.py` is **fixed**. It is the scorer. It counts tokens itself through `Meter`, and it fixes the model (qwen3-8b-8k), the context size (8192) and the call cap (20 per task).
- `src/leanloop/lean.py` is **the only file you edit**. That covers the system prompt, the tool schemas, the paging rules, the step budget and the output cap.
- `research/results.tsv` is the experiment log. `research/snapshots/` holds a copy of `lean.py` from every experiment.

## Goal
Score on `dev` with `python research/evaluate.py lean dev`, accuracy first, then tokens:
1. Higher `correct` is better.
2. When `correct` is equal, lower `tokens` is better. Count it as a real gain only if it's at least 3% lower.

**Keep** a change if it raises `correct`, or if it keeps `correct` the same and passes the 3% rule. Otherwise **discard** it and put back the best `lean.py`.

## Rules
- Use general harness ideas only. Don't add hints about these particular tasks, files, column names or answers, and nothing that only makes sense for this workspace. The held-out `test` set will catch any overfitting.
- Don't change how tokens are counted. Don't send output around the model, and don't hide the prompt.
- Simplicity matters. A small gain that adds a lot of complex code isn't worth keeping. Removing code and getting the same or a better score counts as a win.
- The first run is always the unchanged baseline.
- Run `test` only at the end, for the final comparison against pi, naive and lean0.

## Loop
1. Change `lean.py` to try one idea.
2. Run `python research/evaluate.py lean dev > research/run.log 2>&1`.
3. Read the summary lines. If the run crashed, fix it once or twice; if it still fails, log it as `crash`.
4. Add a row to `results.tsv` with `exp, correct, tokens, calls, seconds, status, description`.
5. Keep or discard according to the Goal rule.
