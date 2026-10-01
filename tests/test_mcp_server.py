"""The MCP server: the dictionary lookup, and the five tools driven by a real MCP client over
stdio against a server process on the loaded benchmark."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from src.mcp_server import lookup

ROOT = Path(__file__).resolve().parent.parent
TOOLS = {"list_tables", "describe_table", "sample_rows", "run_sql", "lookup_dictionary"}


@pytest.fixture(scope="module")
def dictionary():
    return yaml.safe_load((ROOT / "dictionary/financial.yaml").read_text(encoding="utf-8"))


class TestLookup:
    def test_a_code_is_translated(self, dictionary):
        out = lookup.search(dictionary, "poplatek mesicne")
        assert out["ok"]
        assert any(
            h["kind"] == "code" and h["table"] == "account" and "monthly" in h["text"]
            for h in out["hits"]
        )

    def test_a_column_is_found_by_its_meaning(self, dictionary):
        out = lookup.search(dictionary, "statement frequency")
        assert any(h["kind"] == "column" and h["column"] == "frequency" for h in out["hits"])

    def test_the_quirks_are_searchable(self, dictionary):
        out = lookup.search(dictionary, "koruna")
        assert any(h["kind"] == "quirk" for h in out["hits"])

    def test_hits_are_capped_and_say_so(self, dictionary):
        out = lookup.search(dictionary, "a")
        assert len(out["hits"]) == lookup.MAX_HITS and out["truncated"] and out["matches"] > 15

    def test_empty_and_long_terms_are_refused(self, dictionary):
        assert not lookup.search(dictionary, "  ")["ok"]
        assert not lookup.search(dictionary, "x" * 101)["ok"]

    def test_nothing_found_is_an_ok_empty_result(self, dictionary):
        out = lookup.search(dictionary, "zzzzzz-not-a-term")
        assert out["ok"] and out["hits"] == [] and out["matches"] == 0


def stdio_params(**extra_env: str) -> StdioServerParameters:
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.update(extra_env)
    return StdioServerParameters(
        command=sys.executable, args=["-m", "src.mcp_server"], env=env, cwd=str(ROOT)
    )


def call(tool: str, args: dict | None = None, **env: str):
    """One tool call from a real client against a freshly started server process."""

    async def go():
        async with Client(stdio_params(**env)) as client:
            return await client.call_tool(tool, args or {})

    return asyncio.run(go())


def data(result) -> dict:
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


@pytest.mark.bird
class TestOverStdio:
    def test_the_server_lists_exactly_the_five_read_only_tools(self, bird_ready):
        async def go():
            async with Client(stdio_params()) as client:
                return await client.list_tools()

        tools = asyncio.run(go()).tools
        assert {t.name for t in tools} == TOOLS
        assert all(
            t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools
        )

    def test_the_tools_answer_a_real_question(self, bird_ready):
        async def go():
            async with Client(stdio_params()) as client:
                tables = data(await client.call_tool("list_tables", {}))
                loan = data(await client.call_tool("describe_table", {"table": "loan"}))
                count = data(
                    await client.call_tool("run_sql", {"sql": "SELECT COUNT(*) FROM loan"})
                )
                code = data(
                    await client.call_tool("lookup_dictionary", {"term": "POPLATEK MESICNE"})
                )
                sample = data(await client.call_tool("sample_rows", {"table": "loan", "n": 3}))
            return tables, loan, count, code, sample

        tables, loan, count, code, sample = asyncio.run(go())
        assert len(tables["tables"]) == 8 and loan["table"] == "loan"
        assert count["ok"] and count["rows"] == [[682]]
        assert code["ok"] and any(h["kind"] == "code" for h in code["hits"])
        assert sample["ok"] and sample["row_count"] == 3

    def test_a_write_or_second_statement_is_refused_and_nothing_changes(self, bird_ready):
        for sql in (
            "DELETE FROM loan",
            "SELECT 1; DROP TABLE loan",
            "SELECT * FROM pg_shadow",
            "SELECT pg_sleep(60)",
        ):
            out = data(call("run_sql", {"sql": sql}))
            assert not out["ok"], sql
        assert data(call("run_sql", {"sql": "SELECT COUNT(*) FROM loan"}))["rows"] == [[682]]

    def test_the_row_cap_applies_and_the_total_is_reported(self, bird_ready):
        out = data(call("run_sql", {"sql": "SELECT trans_id FROM trans"}))
        assert out["ok"] and len(out["rows"]) == 100 and out["truncated"]
        assert out["total_rows"] is None or out["total_rows"] > 1_000_000

    def test_bad_arguments_are_errors_not_queries(self, bird_ready):
        for tool, args in (
            ("run_sql", {"sql": 7}),
            ("run_sql", {}),
            ("describe_table", {"table": ["x"]}),
        ):
            result = call(tool, args)
            assert result.is_error or not data(result).get("ok", True), (tool, args)

    def test_a_database_without_a_dictionary_says_so(self, bird_ready):
        # the server over a person's database (here the compose role on a schema it can read)
        env = {
            "ANALYST_MCP_PG_HOST": os.environ.get("ANALYST_DB_HOST") or "127.0.0.1",
            "ANALYST_MCP_PG_PORT": os.environ.get("ANALYST_DB_PORT") or "5432",
            "ANALYST_MCP_PG_DBNAME": "bird",
            "ANALYST_MCP_PG_USER": "analyst_ro",
            "ANALYST_MCP_PG_PASSWORD": os.environ.get("ANALYST_RO_PASSWORD") or "analyst-local-ro",
            "ANALYST_MCP_PG_SCHEMA": "financial",
        }
        out = data(call("lookup_dictionary", {"term": "loan"}, **env))
        assert not out["ok"] and out["error"]["kind"] == "unavailable"
        assert data(call("run_sql", {"sql": "SELECT COUNT(*) FROM loan"}, **env))["rows"] == [[682]]

    def test_a_slow_query_does_not_stall_the_protocol(self, bird_ready):
        import time

        slow = "SELECT COUNT(*) FROM trans a JOIN trans b ON a.account_id = b.account_id"

        async def go():
            async with Client(stdio_params()) as client:
                running = asyncio.create_task(client.call_tool("run_sql", {"sql": slow}))
                await asyncio.sleep(1.0)
                started = time.perf_counter()
                await client.list_tools()
                answered_in = time.perf_counter() - started
                await running
                return answered_in

        assert asyncio.run(go()) < 5.0
