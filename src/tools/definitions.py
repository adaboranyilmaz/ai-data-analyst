"""The tools' names, descriptions and input schemas, in the form the Anthropic Messages API and
MCP both take (`name`, `description`, `input_schema`).

The database is not an input: the caller binds the tools to one database, so the model can
neither choose nor name another. `validate_chart` checks a spec against the columns of the
result it will draw, which the caller supplies from the query the chart belongs to. The
descriptions are provisional until the agent's prompts are frozen.
"""

from __future__ import annotations

TOOLS: list[dict] = [
    {
        "name": "list_tables",
        "description": (
            "List the tables of the database, with their row counts and, where the data "
            "dictionary has them, their names and descriptions."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "describe_table",
        "description": (
            "Describe one table: its columns with their types and profiles (nulls, distinct "
            "values, range, and the values of low-cardinality text columns), and the data "
            "dictionary's meaning of the table and each column, its codes and its join paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"table": {"type": "string", "description": "The table's name."}},
            "required": ["table"],
            "additionalProperties": False,
        },
    },
    {
        "name": "sample_rows",
        "description": "Show a few whole rows of one table (the same rows each time).",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "The table's name."},
                "n": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "How many rows (default 5).",
                },
            },
            "required": ["table"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_sql",
        "description": (
            "Run one read-only PostgreSQL SELECT query (WITH allowed) and return its columns, "
            "the first 100 rows and the total row count. Refused: anything but a single "
            "query, comments, system catalogs, and joins without ON or USING. Long text "
            "values are shortened. A query is stopped after 15 seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "The query."}},
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "validate_chart",
        "description": (
            "Check a Vega-Lite (v6) chart spec for the result of the last query: valid "
            'Vega-Lite, data given as {"name": "result"}, and only fields that are columns '
            "of that result. No URLs or links."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"spec": {"type": "object", "description": "The Vega-Lite spec."}},
            "required": ["spec"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOLS)
