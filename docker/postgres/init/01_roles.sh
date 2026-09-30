#!/bin/bash
# The agent's read-only role, and a canary table the role tests try to change. Runs once, when
# the data volume is first initialized (the image's /docker-entrypoint-initdb.d hook), as the
# admin superuser. The entrypoint executes this file when it is executable and sources it
# otherwise (a checkout without the executable bit), so it sets no shell options of its own:
# the entrypoint already stops on the first failing command, and psql's exit status is this
# script's.
: "${ANALYST_RO_PASSWORD:?ANALYST_RO_PASSWORD must be set}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v ro_password="$ANALYST_RO_PASSWORD" <<'EOSQL'
-- The agent's role: it can log in and read, nothing else.
CREATE ROLE analyst_ro LOGIN PASSWORD :'ro_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS
    CONNECTION LIMIT 20;

-- Session defaults. default_transaction_read_only and statement_timeout are defaults the
-- role can change in its own session, so they are a convenience, not a guard: the
-- privileges below are what stop writes. temp_file_limit can only be changed by a
-- superuser, so it holds even against the role's own SQL.
ALTER ROLE analyst_ro SET default_transaction_read_only = on;
ALTER ROLE analyst_ro SET statement_timeout = '30s';
ALTER ROLE analyst_ro SET idle_in_transaction_session_timeout = '60s';
ALTER ROLE analyst_ro SET temp_file_limit = '256MB';

-- Nothing for PUBLIC: no CONNECT or TEMPORARY on the database, and no CREATE or USAGE in
-- the public schema. The agent gets CONNECT, and USAGE and SELECT where it is granted below.
REVOKE ALL ON DATABASE :"DBNAME" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"DBNAME" TO analyst_ro;
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- A table the agent can read and must never be able to change (tests/test_db_role.py).
CREATE SCHEMA role_check;
CREATE TABLE role_check.canary (id integer PRIMARY KEY, note text NOT NULL);
INSERT INTO role_check.canary VALUES (1, 'must never change');
GRANT USAGE ON SCHEMA role_check TO analyst_ro;
GRANT SELECT ON role_check.canary TO analyst_ro;
EOSQL
