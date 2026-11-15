"""IR micro-span pruning: keep only the sentences of a document that are relevant to the question.

Pipeline (all deterministic, no LLM tokens):
  1. split text into micro-spans (1-2 sentences)
  2. [guard] drop spans that look like instructions aimed at the model (prompt-injection patterns)
  3. embed the question and every span (local Ollama `nomic-embed-text` by default), rank by cosine similarity
  4. keep the top-k spans above a similarity floor, in their original order

Embedding tokens are NOT free (a hosted embedder bills for them), so every call reports how many it used.
`embed` is injectable: tests use a fake one, and a Cohere Embed/Rerank client can be swapped in.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from array import array
from dataclasses import dataclass, field

from . import safety

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
QUERY_PREFIX, DOC_PREFIX = "search_query: ", "search_document: "

SPAN_SENTENCES = 2     # sentences per micro-span
MAX_SPAN_CHARS = 400
TOP_K = 5
MIN_SIM = 0.45         # similarity floor; chosen a priori, not tuned on the benchmark
EMBED_BATCH = 64       # texts per request, so a big corpus is many small calls, not one giant one
EMBED_MAX_CHARS = 6000 # longer texts are cut before embedding (the model's context is ~2k tokens)
CACHE_MAX = 20000      # embedding cache entries (oldest evicted first)

# Instruction-like text aimed at a model, not at a reader. Deterministic, zero tokens.
# Best effort: it can be dodged by paraphrase, and it can flag a legitimate sentence such as "Reply with your badge
# number" -- which is why dropped spans are reported to the user instead of vanishing silently.
INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any)\b[^.\n]{0,30}\b(instruction|prompt|rule|direction)s?"
    r"|\b(system|assistant|developer)\s*(prompt|message|override|note)?\s*:"
    r"|\b(you (must|should|will|are to)|always (reply|respond|answer))\b[^.\n]{0,60}\b(reply|respond|answer|say|output|print|write)\b"
    r"|\b(reply|respond|answer|output) (only )?with\b"
    r"|\bdo not (tell|mention|reveal)\b[^.\n]{0,30}\b(user|anyone)\b"
    r"|\b(new|updated) instructions?\b",
    re.I)


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(safety.normalize(text)))


@dataclass
class Pruned:
    spans: list[str]                       # kept spans, original order
    embed_tokens: int = 0
    dropped_guard: int = 0                 # spans removed by the injection guard
    total_spans: int = 0
    scores: list[float] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)   # the spans the guard removed (for display)


def _windows(s: str, n: int = MAX_SPAN_CHARS) -> list[str]:
    """Cut an over-long sentence into pieces of at most n chars, preferring spaces. Nothing is thrown away."""
    out = []
    while len(s) > n:
        cut = s.rfind(" ", 0, n)
        cut = cut if cut > n // 2 else n
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    return out + ([s] if s else [])


def split_spans(text: str, sentences: int = SPAN_SENTENCES) -> list[str]:
    """Split into micro-spans of up to `sentences` sentences (and at most MAX_SPAN_CHARS).
    Lines break spans (lists, headings, log lines); very long lines are windowed, not truncated."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [w for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", line) if p.strip() for w in _windows(p.strip())]
        cur: list[str] = []
        for p in parts:
            if cur and (len(cur) >= sentences or len(" ".join(cur + [p])) > MAX_SPAN_CHARS):
                out.append(" ".join(cur))
                cur = []
            cur.append(p)
        if cur:
            out.append(" ".join(cur))
    return out


def ollama_embed(texts: list[str], model: str = EMBED_MODEL, timeout: int = 300) -> tuple[list[list[float]], int]:
    """Embed texts with a local Ollama model, in batches. Returns (vectors, tokens used)."""
    vecs, tokens = [], 0
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = [t[:EMBED_MAX_CHARS] for t in texts[i:i + EMBED_BATCH]]
        req = urllib.request.Request(f"{OLLAMA}/api/embed", data=json.dumps({"model": model, "input": chunk}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        if len(d.get("embeddings", [])) != len(chunk):
            raise ValueError(f"embedding server returned {len(d.get('embeddings', []))} vectors for {len(chunk)} texts")
        vecs += d["embeddings"]
        tokens += int(d.get("prompt_eval_count", 0))
    return vecs, tokens


def _cos(a, b) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def prune(question: str, spans: list[str], *, embed=ollama_embed, k: int = TOP_K, min_sim: float = MIN_SIM,
          guard: bool = True) -> Pruned:
    """Rank `spans` against `question`; keep the top-k above `min_sim` (always at least the single best span)."""
    total, k = len(spans), max(1, k)
    dropped: list[str] = []
    if guard:
        keep = []
        for s in spans:
            (dropped if looks_like_injection(s) else keep).append(s)
        spans = keep
    if not spans:
        return Pruned([], 0, len(dropped), total, dropped=dropped)
    vecs, tokens = embed([QUERY_PREFIX + question[:EMBED_MAX_CHARS]] + [DOC_PREFIX + s for s in spans])
    q, docs = vecs[0], vecs[1:]
    scored = sorted(((_cos(q, v), i) for i, v in enumerate(docs)), reverse=True)
    picked = [(sc, i) for sc, i in scored[:k] if sc >= min_sim] or scored[:1]
    picked.sort(key=lambda p: p[1])       # restore document order
    return Pruned([spans[i] for _, i in picked], tokens, len(dropped), total, [round(sc, 3) for sc, _ in picked], dropped)


class EmbedCache:
    """Memoizing embedder. Documents are embedded once (a one-time 'index' cost); each question pays only for itself.
    Counters: index_tokens (cache misses on documents) and query_tokens (questions).
    Bounded (CACHE_MAX entries, oldest evicted) and compact (float32 arrays), so a long session cannot eat memory."""

    def __init__(self, embed=ollama_embed, max_entries: int = CACHE_MAX):
        self.embed, self.cache, self.max_entries = embed, {}, max_entries
        self.index_tokens = self.query_tokens = 0

    def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        used = 0
        for group, is_query in (([t for t in texts if t.startswith(QUERY_PREFIX)], True),
                                ([t for t in texts if not t.startswith(QUERY_PREFIX)], False)):
            miss = list(dict.fromkeys(t for t in group if t not in self.cache))
            if miss:
                vecs, tok = self.embed(miss)
                if len(vecs) != len(miss):
                    raise ValueError("embedder returned the wrong number of vectors")
                self.cache.update((t, array("f", v)) for t, v in zip(miss, vecs))
                used += tok
                if is_query:
                    self.query_tokens += tok
                else:
                    self.index_tokens += tok
        result = [self.cache[t] for t in texts]           # build the answer first, THEN evict
        while len(self.cache) > self.max_entries:
            self.cache.pop(next(iter(self.cache)))
        return result, used


# ----------------------------------------------------------------------------- conflicting sources

_NUM = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
_WORD = re.compile(r"[a-z]{4,}")
_FILLER = {"with", "that", "this", "from", "have", "been", "will", "your", "their", "there", "which", "also", "must", "should",
           "than", "then", "into", "such", "each", "only", "more", "most", "over", "after", "before", "about", "note", "notice",
           "correction", "update", "updated", "now", "currently"}
CONFLICT_MIN_SHARED = 3        # topic words two sentences must share to count as "about the same thing"
CONFLICT_MIN_OVERLAP = 0.6     # ... and that share must be at least this fraction of the smaller sentence's topic words
TEMPLATE_DF = 6                # a word found in more than this many sentences of the pool is boilerplate, not a topic word


def _facts(span: str) -> tuple[set[str], set[str]]:
    return set(_WORD.findall(span.lower())) - _FILLER, {n.replace(",", "").lstrip("$") for n in _NUM.findall(span)}


def find_conflicts(spans: list[str], pool: list[str] | None = None) -> list[tuple[str, str]]:
    """Pairs of `spans` that talk about the same thing but give different numbers ("refund within 21 days" vs "within 3 days").
    Deterministic, zero tokens. `pool` is every sentence the spans were drawn from: words that are all over the pool
    ("every", "once", "days" in boilerplate like "X is cleaned once every N days") are ignored, so only real topic words count.
    It cannot tell WHICH statement is right (a planted correction and a real one look alike), so the caller should show both to
    the user rather than silently choose. It will flag some legitimate differences; that is the safe direction for a safety check."""
    pool = pool or spans
    df: dict[str, int] = {}
    for sp in pool:
        for w in _facts(sp)[0]:
            df[w] = df.get(w, 0) + 1
    facts = [({w for w in ws if df.get(w, 0) <= TEMPLATE_DF}, ns) for ws, ns in map(_facts, spans)]
    out = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            (wi, ni), (wj, nj) = facts[i], facts[j]
            shared, small = wi & wj, min(len(wi), len(wj))
            if (ni and nj and ni != nj and not (ni <= nj or nj <= ni) and len(shared) >= CONFLICT_MIN_SHARED
                    and small and len(shared) / small >= CONFLICT_MIN_OVERLAP):
                out.append((spans[i], spans[j]))
    return out
