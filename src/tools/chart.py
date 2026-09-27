"""`validate_chart`: checks a Vega-Lite chart spec before any browser renders it.

A spec passes if it is valid against the official Vega-Lite 6.4.3 JSON schema (vendored,
gzipped, and checked against its pinned hash when loaded), if every field it references is a
column of the query result (or a field the chart spec's own transforms create), and if it can make
the viewer's browser load nothing: its only data is the query result, named `result` and
filled in by the page that renders it. So a spec may not contain a `url` anywhere (data from
a URL, or an image mark's source), an `href` (a link opened on click), `usermeta`, or its own
`datasets`. Without these rules a chart could send the viewer's browser to any address, with
query results in the request.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import best_match

SCHEMA_FILE = Path(__file__).parent / "schemas/vega-lite-v6.4.3.schema.json.gz"
# sha256 of the uncompressed schema as published at
# https://cdn.jsdelivr.net/npm/vega-lite@6.4.3/build/vega-lite-schema.json
SCHEMA_SHA256 = "4f11cd379b7cac0ddee17eefea84c028bd41619ace28778acf843c009e43abd2"
SCHEMA_URLS = frozenset(
    {
        "https://vega.github.io/schema/vega-lite/v6.json",
        "https://vega.github.io/schema/vega-lite/v6.4.3.json",
    }
)
DATA_NAME = "result"
DENIED_KEYS = frozenset({"url", "href", "usermeta", "datasets"})
MAX_ERRORS = 5


@functools.cache
def _validator() -> Draft7Validator:
    raw = gzip.decompress(SCHEMA_FILE.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SCHEMA_SHA256:
        raise RuntimeError(f"{SCHEMA_FILE.name}: sha256 {digest} is not the pinned schema's")
    return Draft7Validator(json.loads(raw))


def _walk(node: Any, path: tuple = ()):
    """Every (path, key, value) of every object in the chart spec."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield path, k, v
            yield from _walk(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, (*path, i))


def _where(path: tuple, key: str) -> str:
    return "/".join(str(p) for p in (*path, key))


def _strings(v: Any) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


def _unescape(field: str) -> str:
    return re.sub(r"\\(.)", r"\1", field)


def validate_chart(spec: Any, columns: list[str], max_spec_chars: int = 20_000) -> dict:
    """`columns`: the names of the result's columns, the only data the chart can show."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return {"ok": False, "errors": ["the chart spec must be a JSON object"]}
    if len(json.dumps(spec)) > max_spec_chars:
        return {
            "ok": False,
            "errors": [f"the chart spec is longer than {max_spec_chars:,} characters"],
        }

    if "$schema" in spec and spec["$schema"] not in SCHEMA_URLS:
        errors.append(f"$schema must be {sorted(SCHEMA_URLS)[0]}")
    declared: set[str] = set()
    references: list[tuple[str, str]] = []
    for path, key, value in _walk(spec):
        if key in DENIED_KEYS:
            errors.append(f"{_where(path, key)}: {key!r} is not allowed")
        elif key == "data" and value != {"name": DATA_NAME}:
            errors.append(f'{_where(path, key)}: data must be {{"name": "{DATA_NAME}"}}')
        elif key == "mark" and (
            value == "image" or (isinstance(value, dict) and value.get("type") == "image")
        ):
            errors.append(f"{_where(path, key)}: image marks are not allowed")
        elif key == "as":
            declared.update(_strings(value))
        elif key == "field" and isinstance(value, str):
            references.append((_where(path, key), value))
        elif key in ("groupby", "fold", "fields"):
            references += [(_where(path, key), f) for f in _strings(value)]

    known = set(columns) | declared
    for where, field in references:
        name = _unescape(field)
        if name not in known:
            errors.append(f"{where}: field {field!r} is not a column of the result")
        elif re.search(r"(?<!\\)[.\[]", field):
            errors.append(
                f"{where}: field {field!r}: escape '.' and '[' in field names with a backslash"
            )

    schema_errors = list(_validator().iter_errors(spec))
    if schema_errors:
        best = best_match(schema_errors)
        where = "/".join(str(p) for p in best.absolute_path) or "(top level)"
        errors.append(f"not valid Vega-Lite at {where}: {best.message[:300]}")
    return {
        "ok": not errors,
        "errors": errors[:MAX_ERRORS]
        + ([f"... and {len(errors) - MAX_ERRORS} more"] if len(errors) > MAX_ERRORS else []),
    }
