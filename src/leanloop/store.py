"""Off-prompt store: the 'disk' in leanloop's paging idea. Big outputs live here, not in the prompt."""

from __future__ import annotations

import re


class Store:
    def __init__(self):
        self._items: dict[int, str] = {}

    def put(self, text: str) -> int:
        h = len(self._items) + 1
        self._items[h] = text
        return h

    def get(self, h: int) -> str | None:
        return self._items.get(h)

    def fetch(self, h: int, start: int = 1, end: int | None = None, max_chars: int = 2000) -> str:
        text = self._items.get(h)
        if text is None:
            return f"error: no stored output #{h}"
        lines = text.splitlines()
        start = max(1, int(start))
        end = min(len(lines), int(end) if end else start + 39)
        out = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n[lines {start}-{end} of {len(lines)}]"
        return (out[:max_chars] + "\n[cut]" if len(out) > max_chars else out) + more

    def search(self, h: int, pattern: str, max_hits: int = 15) -> str:
        text = self._items.get(h)
        if text is None:
            return f"error: no stored output #{h}"
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"error: bad pattern ({e})"
        hits = [f"{i}: {ln}" for i, ln in enumerate(text.splitlines(), 1) if rx.search(ln)]
        shown = "\n".join(h_[:200] for h_ in hits[:max_hits])
        return f"{shown}\n[{len(hits)} matching lines, showing {min(len(hits), max_hits)}]"
