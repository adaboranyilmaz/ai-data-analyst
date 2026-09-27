"""The query guard: the first of two independent layers between the agent's SQL and the data.

`QueryGuard.check` accepts a query only if it is a single read-only query over the tables of
one schema, and returns every reason it has for refusing one, so the agent can correct its SQL.
The second layer is the database itself (the agent's role, its privileges and the read-only
execution in src/db/execute.py); each layer is tested to stop every attack in the security
suite on its own.

Two passes:

1. **Lexical, written for PostgreSQL's own rules.** A parser in another language decides where
   a string or comment ends; if it disagrees with PostgreSQL, text that looks like a string to
   the checker can run as SQL on the server. So everything whose reading depends on such
   rules is refused outright: comments, backslashes, `$` (dollar quoting and parameters),
   `E''` and `U&` escapes, a second statement, and any character outside printable ASCII
   except inside a string or a quoted name, where invisible and control characters are still
   refused. What remains is quoted with `'...'` and `"..."` only, where both readers agree.
   On the text outside the quotes, every name is checked: nothing starting with `pg_`
   (system catalogs and functions), no `information_schema`, no object-identifier types
   (`regclass` and the like, which look objects up by name), and no call to a function on the
   deny list.
2. **Structural, on sqlglot's syntax tree.** The statement is a query (`SELECT`, a set
   operation, or either under `WITH`), with no `WITH RECURSIVE`, `SELECT INTO`, `FOR UPDATE`
   or `LATERAL`, no data-modifying statement anywhere, and every table in the allowed set
   (or a CTE). Joins name their condition: no `CROSS JOIN`, `NATURAL JOIN` or comma join.
   `generate_series`, `repeat`, `lpad` and `rpad` take literal sizes under a bound.

What the guard cannot judge is cost: a query that joins large tables on a legitimate
condition can still run long. The execution layer's time and row limits contain that.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

# sqlglot logs a warning when it falls back to parsing a statement as an opaque command; the
# guard refuses such statements and says so in its verdict, so the warning is noise.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

MAX_QUERY_CHARS = 20_000
MAX_SERIES_VALUES = 100_000
MAX_STRING_LENGTH = 10_000

# Functions refused by name, whatever sqlglot makes of them. With `pg_` names refused
# everywhere, these cover the rest of what reads the server's state, changes it or runs SQL
# given as text.
DENIED_FUNCTIONS = frozenset(
    {
        "set_config",
        "current_setting",
        "version",
        "nextval",
        "setval",
        "currval",
        "lastval",
        "loread",
        "lowrite",
        "query_to_xml",
        "query_to_xmlschema",
        "query_to_xml_and_xmlschema",
        "cursor_to_xml",
        "cursor_to_xmlschema",
        "current_database",
        "current_schema",
        "current_schemas",
        "current_user",
        "session_user",
        "current_role",
        "inet_server_addr",
        "inet_server_port",
        "inet_client_addr",
        "inet_client_port",
        "has_table_privilege",
        "has_schema_privilege",
        "has_function_privilege",
        "has_database_privilege",
        "has_column_privilege",
        "to_regclass",
        "to_regproc",
        "to_regprocedure",
        "to_regtype",
        "to_regrole",
        "to_regnamespace",
        "obj_description",
        "col_description",
        "shobj_description",
        "format_type",
        "dblink",
    }
)
DENIED_FUNCTION_PREFIXES = ("pg_", "lo_", "txid_", "dblink_", "has_", "to_reg")
DENIED_FUNCTION_SUBSTRINGS = ("_to_xml",)
OBJECT_IDENTIFIER_TYPES = frozenset(
    {
        "oid",
        "regclass",
        "regproc",
        "regprocedure",
        "regoper",
        "regoperator",
        "regtype",
        "regrole",
        "regnamespace",
        "regconfig",
        "regdictionary",
        "regcollation",
    }
)
DENIED_NAMES = frozenset({"information_schema"}) | OBJECT_IDENTIFIER_TYPES

_WORD = re.compile(r'"((?:[^"]|"")*)"|([A-Za-z_][A-Za-z0-9_]*)')
_CALL = re.compile(r'(?:"((?:[^"]|"")*)"|([A-Za-z_][A-Za-z0-9_]*))\s*\(')
_STATEMENT_NODES = tuple(
    getattr(exp, n)
    for n in (
        "DML",
        "DDL",
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Set",
        "Copy",
        "TruncateTable",
        "Transaction",
        "Commit",
        "Rollback",
        "Grant",
    )
    if hasattr(exp, n)
)


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reasons: tuple[str, ...] = ()
    query: str = ""  # the query to execute: the input without a trailing semicolon


@dataclass
class _Scan:
    code: str = ""  # the query with every string literal emptied; quoted names kept
    problems: list[str] = field(default_factory=list)
    query: str = ""


def _char_problem(c: str, where: str) -> str | None:
    cat = unicodedata.category(c)
    if (cat == "Cc" and c not in "\t\n\r") or cat in ("Cf", "Co", "Cs", "Cn", "Zl", "Zp"):
        return f"invisible or control character U+{ord(c):04X} {where}"
    return None


def _comment_end(query: str, i: int) -> int:
    """The index after the comment starting at i; block comments nest, as in PostgreSQL."""
    if query.startswith("--", i):
        j = query.find("\n", i)
        return len(query) if j < 0 else j
    depth, j = 0, i
    while j < len(query):
        if query.startswith("/*", j):
            depth, j = depth + 1, j + 2
        elif query.startswith("*/", j):
            depth, j = depth - 1, j + 2
            if depth == 0:
                return j
        else:
            j += 1
    return len(query)


def scan(query: str) -> _Scan:
    """The lexical pass (see the module docstring)."""
    out = _Scan()
    code: list[str] = []
    problems: list[str] = []

    def problem(p: str) -> None:
        if p not in problems:
            problems.append(p)

    i, n = 0, len(query)
    end = n
    while i < n:
        c = query[i]
        if c in "'\"":
            prev = query[max(0, i - 2) : i]
            if c == "'" and re.search(r"(?:^|[^A-Za-z0-9_])[Ee]$", prev):
                problem("E'' strings (backslash escapes) are not allowed")
            if prev.upper() == "U&":
                problem("U& escapes are not allowed")
            j = i + 1
            while True:
                if j >= n:
                    problem("unterminated quoted string or name")
                    break
                if query[j] == c:
                    if j + 1 < n and query[j + 1] == c:  # doubled quote: a literal quote
                        j += 2
                        continue
                    break
                if query[j] == "\\":
                    problem("backslashes are not allowed")
                elif p := _char_problem(query[j], "inside a quoted string or name"):
                    problem(p)
                j += 1
            code.append("''" if c == "'" else query[i : j + 1])
            i = j + 1
            continue
        if query.startswith("--", i) or query.startswith("/*", i):
            # refused; skipped as PostgreSQL would skip it, so the reasons that follow are
            # about the text PostgreSQL would run
            problem("comments are not allowed")
            i = _comment_end(query, i)
            code.append(" ")
            continue
        if c == "$":
            problem("$ is not allowed (dollar quoting or parameters)")
        elif c == "\\":
            problem("backslashes are not allowed")
        elif c == ";":
            if query[i + 1 :].strip(" \t\r\n;"):
                problem("only one statement is allowed")
            else:
                end = i
                break
        elif not (" " <= c <= "~" or c in "\t\r\n"):
            problem(
                _char_problem(c, "outside a quoted string")
                or f"character U+{ord(c):04X} outside a quoted string (only ASCII is allowed)"
            )
        code.append(c)
        i += 1
    out.code = "".join(code)
    out.query = query[:end].rstrip()

    for m in _WORD.finditer(out.code):
        word = (m.group(1) if m.group(1) is not None else m.group(2)).lower()
        if word.startswith("pg_"):
            problem(f"system objects are not allowed ({word})")
        elif word in DENIED_NAMES:
            problem(f"{word} is not allowed")
    for m in _CALL.finditer(out.code):
        name = (m.group(1) if m.group(1) is not None else m.group(2)).lower()
        if (
            name in DENIED_FUNCTIONS
            or name.startswith(DENIED_FUNCTION_PREFIXES)
            or any(s in name for s in DENIED_FUNCTION_SUBSTRINGS)
        ):
            problem(f"function {name}() is not allowed")
    out.problems = problems
    return out


def _int_literal(node: exp.Expression | None) -> int | None:
    if isinstance(node, exp.Neg):
        v = _int_literal(node.this)
        return -v if v is not None else None
    if isinstance(node, exp.Paren):
        return _int_literal(node.this)
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(node.this)
        except ValueError:
            return None
    return None


def _size_problems(tree: exp.Expression) -> list[str]:
    problems = []
    series = tuple(
        getattr(exp, n) for n in ("GenerateSeries", "ExplodingGenerateSeries") if hasattr(exp, n)
    )
    for node in tree.find_all(*series):
        start, stop = _int_literal(node.args.get("start")), _int_literal(node.args.get("end"))
        step = node.args.get("step")
        step_v = _int_literal(step) if step is not None else 1
        if start is None or stop is None or not step_v:
            problems.append("generate_series() needs whole-number literal bounds and step")
        elif abs(stop - start) // abs(step_v) + 1 > MAX_SERIES_VALUES:
            problems.append(f"generate_series() is limited to {MAX_SERIES_VALUES:,} values")
    for node in tree.find_all(exp.Repeat):
        times = _int_literal(node.args.get("times"))
        if times is None or times > MAX_STRING_LENGTH:
            problems.append(f"repeat() needs a literal count of at most {MAX_STRING_LENGTH:,}")
    for node in tree.find_all(exp.Pad):
        length = _int_literal(node.args.get("expression"))
        if length is None or length > MAX_STRING_LENGTH:
            problems.append(f"lpad()/rpad() need a literal length of at most {MAX_STRING_LENGTH:,}")
    for node in tree.find_all(exp.Anonymous):  # in case sqlglot leaves one of them untyped
        if node.name.lower() in ("generate_series", "repeat", "lpad", "rpad"):
            problems.append(f"{node.name.lower()}() could not be checked")
    return problems


def _structure_problems(tree: exp.Expression, schema: str, tables: frozenset[str]) -> list[str]:
    problems: list[str] = []
    root = tree.this if isinstance(tree, exp.Subquery) else tree
    if not isinstance(root, exp.Select | exp.SetOperation):
        return [f"only a SELECT query is allowed, not {type(root).__name__.upper()}"]
    for node in tree.walk():
        if isinstance(node, _STATEMENT_NODES):
            problems.append(f"{type(node).__name__.upper()} is not allowed inside a query")
        elif isinstance(node, exp.Into):
            problems.append("SELECT INTO is not allowed")
        elif isinstance(node, exp.Lock):
            problems.append("FOR UPDATE / FOR SHARE is not allowed")
        elif isinstance(node, exp.Lateral):
            problems.append("LATERAL is not allowed")
        elif isinstance(node, exp.With) and node.args.get("recursive"):
            problems.append("WITH RECURSIVE is not allowed")
        elif isinstance(node, exp.DataType) and (
            node.sql(dialect="postgres").lower() in OBJECT_IDENTIFIER_TYPES
        ):
            problems.append(f"type {node.sql(dialect='postgres').lower()} is not allowed")
        elif isinstance(node, exp.Join):
            if (node.args.get("kind") or "").upper() == "CROSS":
                problems.append("CROSS JOIN is not allowed; join on a condition")
            elif (node.args.get("method") or "").upper() == "NATURAL":
                problems.append("NATURAL JOIN is not allowed; join on a named condition")
            elif not node.args.get("on") and not node.args.get("using"):
                problems.append("every join needs ON or USING (no comma joins)")

    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if not t.name:  # a function in FROM, checked with the functions
            continue
        ident = t.this
        name = t.name if isinstance(ident, exp.Identifier) and ident.quoted else t.name.lower()
        if t.catalog:
            problems.append(f"table {t.sql(dialect='postgres')}: no database qualifier allowed")
        elif t.db and t.db.lower() != schema.lower():
            problems.append(f"table {t.sql(dialect='postgres')}: only tables of {schema}")
        elif not t.db and name.lower() in ctes:
            continue
        elif name not in tables:
            problems.append(f"unknown table {t.name!r}")
    return problems + _size_problems(tree)


class QueryGuard:
    """Checks queries against one schema and the tables the agent may read in it."""

    def __init__(self, schema: str, tables: set[str] | frozenset[str]):
        self.schema = schema
        self.tables = frozenset(tables)

    @classmethod
    def for_benchmark_db(cls, db: str, snapshot_dir: Path | None = None) -> QueryGuard:
        """A BIRD database: its schema, and the tables in its committed schema snapshot."""
        from src.dictionary import model

        snap = model.load_snapshot(db, snapshot_dir or model.SNAPSHOT_DIR)
        return cls(db, set(snap["tables"]))

    def check(self, query: str) -> Verdict:
        if len(query) > MAX_QUERY_CHARS:
            return Verdict(False, (f"the query is longer than {MAX_QUERY_CHARS:,} characters",))
        s = scan(query)
        if not s.query.strip():
            return Verdict(False, ("the query is empty",))
        if s.problems:
            return Verdict(False, tuple(s.problems))
        try:
            trees = [t for t in sqlglot.parse(s.query, read="postgres") if t is not None]
        except (ParseError, TokenError) as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            return Verdict(False, (f"the query could not be parsed: {first}",))
        if len(trees) != 1:
            return Verdict(False, ("only one statement is allowed",))
        problems = _structure_problems(trees[0], self.schema, self.tables)
        if problems:
            return Verdict(False, tuple(dict.fromkeys(problems)))
        return Verdict(True, (), s.query)
