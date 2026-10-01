"""`lookup_dictionary`: search a database's data dictionary for a term.

The dictionary says what each table and column means, what the stored codes translate to and
what is known to surprise a reader. A client that only sees column names cannot know that
`POPLATEK MESICNE` means monthly statements; this is how it finds out.
"""

from __future__ import annotations

from typing import Any

MAX_HITS = 15
MAX_TERM_CHARS = 100


def _hit(kind: str, text: str, **where: str) -> dict[str, Any]:
    return {"kind": kind, **where, "text": text}


def search(dictionary: dict[str, Any], term: str) -> dict[str, Any]:
    """Entries of the dictionary that mention `term` (case-insensitive), at most MAX_HITS."""
    term = term.strip()
    if not term:
        return {"ok": False, "error": {"kind": "refused", "reasons": ["the search term is empty"]}}
    if len(term) > MAX_TERM_CHARS:
        return {
            "ok": False,
            "error": {
                "kind": "refused",
                "reasons": [f"the search term is longer than {MAX_TERM_CHARS} characters"],
            },
        }
    needle = term.lower()

    def has(*parts: Any) -> bool:
        return any(needle in str(p).lower() for p in parts if p)

    hits: list[dict[str, Any]] = []
    for code, meaning in (dictionary.get("glossary") or {}).items():
        if has(code, meaning):
            hits.append(_hit("glossary", f"{code}: {meaning}"))
    for quirk in dictionary.get("quirks") or []:
        if has(quirk):
            hits.append(_hit("quirk", " ".join(str(quirk).split())))
    for table, entry in (dictionary.get("tables") or {}).items():
        if has(table, entry.get("name"), entry.get("description")):
            hits.append(
                _hit(
                    "table",
                    f"{entry.get('name', table)}: {entry.get('description', '')}",
                    table=table,
                )
            )
        for column, c in (entry.get("columns") or {}).items():
            codes = c.get("codes") or {}
            if has(column, c.get("name"), c.get("description"), c.get("encoding")):
                hits.append(
                    _hit(
                        "column",
                        " ".join(f"{c.get('name', column)}: {c.get('description', '')}".split()),
                        table=table,
                        column=column,
                    )
                )
            for value, meaning in codes.items():
                if has(value, meaning):
                    hits.append(_hit("code", f"{value} = {meaning}", table=table, column=column))
    return {
        "ok": True,
        "term": term,
        "matches": len(hits),
        "hits": hits[:MAX_HITS],
        "truncated": len(hits) > MAX_HITS,
    }
