"""Pointing the analyst at a person's own database: which roles are accepted, which are refused
and why, and that the tools over an accepted connection keep the guard and the read-only rules.

The tests make a throwaway schema in the compose database with two roles, one that can only read
and one that can also insert; both are dropped afterwards."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from psycopg import sql

from src.db.connection import ADMIN_ROLE, DB_NAME, connect
from src.serving.app import create_app
from src.serving.connections import (
    ConnectionRefused,
    ConnectionSpec,
    ConnectionToolbox,
    parse,
    validate,
)

pytestmark = pytest.mark.db

SCHEMA = "conn_test"
READER, WRITER = "conn_reader", "conn_writer"
PASSWORD = "conn-test-password"  # a throwaway role in the local test database


def spec(user: str = READER, password: str = PASSWORD, schema: str = SCHEMA) -> ConnectionSpec:
    return ConnectionSpec(
        os.environ.get("ANALYST_DB_HOST") or "127.0.0.1",
        int(os.environ.get("ANALYST_DB_PORT") or 5432),
        DB_NAME,
        user,
        password,
        schema,
    )


def drop_roles(conn) -> None:
    for role in (READER, WRITER):
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone():
            conn.execute(f"DROP OWNED BY {role}")  # also revokes its grants, e.g. CONNECT
            conn.execute(f"DROP ROLE {role}")


@pytest.fixture(scope="module")
def own_database(db_ready):
    with connect(ADMIN_ROLE, DB_NAME, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        drop_roles(conn)
        conn.execute(f"CREATE SCHEMA {SCHEMA}")
        conn.execute(f"CREATE TABLE {SCHEMA}.loan (loan_id int primary key, amount int)")
        conn.execute(f"INSERT INTO {SCHEMA}.loan VALUES (1, 100), (2, 250), (3, 900)")
        for role in (READER, WRITER):
            conn.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(PASSWORD)
                )
            )
            conn.execute(f"GRANT CONNECT ON DATABASE {DB_NAME} TO {role}")
            conn.execute(f"GRANT USAGE ON SCHEMA {SCHEMA} TO {role}")
            conn.execute(f"GRANT SELECT ON {SCHEMA}.loan TO {role}")
        conn.execute(f"GRANT INSERT ON {SCHEMA}.loan TO {WRITER}")
    yield
    with connect(ADMIN_ROLE, DB_NAME, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        drop_roles(conn)


class TestValidation:
    def test_a_role_that_can_only_read_is_accepted(self, own_database):
        v = validate(spec())
        assert list(v.tables) == ["loan"]
        assert [c["name"] for c in v.tables["loan"]] == ["loan_id", "amount"]

    def test_a_role_that_can_insert_is_refused_and_told_what_to_change(self, own_database):
        with pytest.raises(ConnectionRefused) as e:
            validate(spec(WRITER))
        assert any(
            "INSERT" in r and "conn_test.loan" in r and "SELECT only" in r for r in e.value.reasons
        )

    def test_a_superuser_is_refused(self, own_database):
        with pytest.raises(ConnectionRefused) as e:
            validate(spec(ADMIN_ROLE, os.environ.get("POSTGRES_PASSWORD") or "analyst-local-admin"))
        assert any("superuser" in r for r in e.value.reasons)

    def test_a_wrong_password_is_refused_without_echoing_it(self, own_database):
        with pytest.raises(ConnectionRefused) as e:
            validate(spec(password="not-the-password-xyz"))
        text = " ".join(e.value.reasons)
        assert "could not connect" in text and "not-the-password-xyz" not in text

    def test_a_schema_with_nothing_readable_is_refused(self, own_database):
        with pytest.raises(ConnectionRefused) as e:
            validate(spec(schema="public"))
        assert "no table or view" in e.value.reasons[0]

    def test_the_password_is_never_in_what_is_shown(self):
        s = spec()
        assert (
            PASSWORD not in repr(s)
            and PASSWORD not in str(s.public())
            and "password" not in s.public()
        )


class TestParse:
    def test_missing_and_bad_fields_are_listed(self):
        with pytest.raises(ConnectionRefused) as e:
            parse({"host": " ", "port": 70000, "schema": "a; DROP TABLE x"})
        text = " ".join(e.value.reasons)
        assert "host" in text and "dbname" in text and "user" in text
        assert "port" in text and "schema" in text

    def test_defaults_are_the_standard_port_and_public_schema(self):
        s = parse({"host": "h", "dbname": "d", "user": "u"})
        assert (s.port, s.schema, s.password) == (5432, "public", "")


class TestToolsOverAConnection:
    def test_the_tools_describe_and_query_the_schema(self, own_database):
        with ConnectionToolbox(validate(spec())) as box:
            assert box.call("list_tables", {})["tables"][0]["table"] == "loan"
            assert box.call("describe_table", {"table": "loan"})["columns"][1]["name"] == "amount"
            out = box.call("run_sql", {"sql": "SELECT SUM(amount) AS s FROM loan"})
            assert out["ok"] and out["rows"] == [[1250]]
            assert box.call("sample_rows", {"table": "loan", "n": 2})["row_count"] == 2

    def test_the_guard_and_the_read_only_transaction_still_apply(self, own_database):
        with ConnectionToolbox(validate(spec())) as box:
            for bad in (
                "INSERT INTO loan VALUES (9, 9)",
                "SELECT 1; DROP TABLE loan",
                "SELECT * FROM pg_class",
                "SELECT * FROM other_schema.loan",
            ):
                assert not box.call("run_sql", {"sql": bad})["ok"], bad
            n = box.call("run_sql", {"sql": "SELECT COUNT(*) FROM loan"})["rows"]
            assert n == [[3]]

    def test_even_a_query_the_guard_passed_cannot_write(self, own_database):
        # the executor's read-only transaction, with the guard bypassed, on the reader's role
        box = ConnectionToolbox(validate(spec()))
        try:
            from src.db.execute import Limits, QueryError

            with pytest.raises(QueryError):
                box.executor.execute(
                    "WITH x AS (INSERT INTO loan VALUES (9, 9) RETURNING 1) SELECT * FROM x",
                    Limits(10, 5),
                )
        finally:
            box.close()
        with connect(ADMIN_ROLE, DB_NAME) as conn:
            assert conn.execute(f"SELECT COUNT(*) FROM {SCHEMA}.loan").fetchone()[0] == 3


class TestConnectionsEndpoint:
    def make(self, tmp_path, local):
        from tests.test_serving_app import Scripted, live_settings

        s = live_settings(tmp_path, Scripted())
        s.local_mode = local
        return TestClient(create_app(s), base_url="http://localhost")

    def body(self, user=READER, password=PASSWORD):
        s = spec(user, password)
        return {
            "host": s.host,
            "port": s.port,
            "dbname": s.dbname,
            "user": s.user,
            "password": s.password,
            "schema": SCHEMA,
        }

    def test_refused_when_the_service_is_not_local(self, own_database, tmp_path):
        r = self.make(tmp_path, local=False).post("/connections", json=self.body())
        assert r.status_code == 403

    def test_a_good_connection_is_registered_without_its_password(self, own_database, tmp_path):
        client = self.make(tmp_path, local=True)
        r = client.post("/connections", json=self.body())
        assert r.status_code == 200 and r.json()["tables"] == 1 and r.json()["calibrated"] is False
        shown = client.get("/connections").text + client.get("/api/meta").text + r.text
        assert PASSWORD not in shown
        assert client.delete("/connections").json() == {"connection": None}

    def test_a_writing_role_is_refused_with_reasons(self, own_database, tmp_path):
        client = self.make(tmp_path, local=True)
        r = client.post("/connections", json=self.body(WRITER))
        assert r.status_code == 422 and r.json()["detail"]["refused"]
        assert client.get("/connections").json() == {"connection": None}

    def test_a_question_over_the_connection_is_answered_without_calibration(
        self, own_database, tmp_path
    ):
        from tests.test_serving_app import events

        client = self.make(tmp_path, local=True)
        client.post("/connections", json=self.body())
        evs = events(client.post("/ask", json={"question": "How many loans?"}))
        confidence = next(e for e in evs if e["type"] == "confidence")
        assert confidence["calibrated"] is None and confidence["not_calibrated"] is True
        assert not any(e["type"] == "error" for e in evs)
        rows = next(e for e in evs if e["type"] == "rows")
        assert rows["ok"] and rows["rows"] == [[3]]


class TestLocalHostsOnly:
    def client(self, tmp_path, local):
        from tests.test_serving_app import Scripted, live_settings

        s = live_settings(tmp_path, Scripted())
        s.local_mode = local
        return TestClient(create_app(s), base_url="http://localhost")

    def test_a_foreign_host_name_is_refused_in_local_mode(self, tmp_path):
        c = self.client(tmp_path, local=True)
        for host in ("evil.example.com", "evil.example.com:8000", "127.0.0.1.evil.example.com"):
            r = c.get("/connections", headers={"Host": host})
            assert r.status_code == 403, host
        assert c.get("/api/meta", headers={"Host": "evil.example.com"}).status_code == 403

    def test_this_machines_names_are_accepted_with_or_without_a_port(self, tmp_path):
        c = self.client(tmp_path, local=True)
        for host in ("localhost", "localhost:8000", "127.0.0.1", "127.0.0.1:8765", "[::1]:8000"):
            assert c.get("/health", headers={"Host": host}).status_code == 200, host

    def test_a_service_that_is_not_local_does_not_filter_hosts(self, tmp_path):
        c = self.client(tmp_path, local=False)
        assert c.get("/health", headers={"Host": "demo.example.com"}).status_code == 200
