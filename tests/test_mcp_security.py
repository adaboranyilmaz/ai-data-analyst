"""The security suite through the MCP path: how a tool result is classified, and the whole suite
against a server process over stdio (no attack may get through, and each leaves no effect)."""

from __future__ import annotations

import pytest

from src.mcp_server import suite


def attack(goal="write"):
    return {"id": "x", "goal": goal}


class TestClassify:
    def test_a_guard_refusal_is_rejected(self):
        out = {"ok": False, "error": {"kind": "refused", "reasons": ["only SELECT"]}}
        assert suite.classify(attack(), out, 0.01, 15)[0] == "rejected"

    def test_an_argument_error_from_the_protocol_is_rejected(self):
        assert (
            suite.classify(attack(), {"ok": False, "protocol_error": "bad"}, 0.01, 15)[0]
            == "rejected"
        )

    def test_a_database_refusal_is_denied_and_a_limit_is_contained(self):
        denied = {"ok": False, "error": {"kind": "permission_denied", "message": "no"}}
        timeout = {"ok": False, "error": {"kind": "timeout", "message": "slow"}}
        assert suite.classify(attack(), denied, 0.01, 15)[0] == "denied"
        assert suite.classify(attack("resource"), timeout, 15.2, 15)[0] == "contained"

    def test_a_harmless_success_is_no_effect_but_a_read_or_names_success_is_a_breach(self):
        ok = {"ok": True, "rows": [[1]]}
        assert suite.classify(attack("write"), ok, 0.01, 15)[0] == "no_effect"
        assert suite.classify(attack("resource"), ok, 0.5, 15)[0] == "contained"
        for goal in ("read", "session", "names"):
            assert suite.classify(attack(goal), ok, 0.01, 15)[0] == "breach"

    def test_running_past_the_limit_is_a_breach_whatever_the_result(self):
        refused = {"ok": False, "error": {"kind": "refused", "reasons": ["x"]}}
        assert suite.classify(attack("resource"), refused, 40.0, 15)[0] == "breach"

    def test_a_tool_error_result_is_read_as_a_protocol_error(self):
        class Result:
            is_error = True
            content = [type("C", (), {"text": "validation error"})()]
            structured_content = None

        assert suite.payload(Result())["protocol_error"] == "validation error"


class TestBoundaryList:
    def test_every_tool_and_an_unknown_one_are_attacked(self):
        tools = {t for _, t, _, _ in suite.BOUNDARY_ATTACKS}
        assert {"run_sql", "describe_table", "sample_rows", "lookup_dictionary"} <= tools
        assert tools - {"run_sql", "describe_table", "sample_rows", "lookup_dictionary"}

    def test_ids_are_unique(self):
        ids = [a[0] for a in suite.BOUNDARY_ATTACKS]
        assert len(ids) == len(set(ids))


@pytest.mark.db
def test_the_whole_suite_through_the_mcp_path_has_no_breach(db_ready):
    result = suite.run_suite()
    assert result["breaches"] == 0 and result["every_attack_blocked"]
    assert result["attacks"] >= 80 and result["boundary_attacks"] == len(suite.BOUNDARY_ATTACKS)
    assert result["injection_row"]["returned_as_a_row_value"]
