-- Database-side hardening for the agent's read-only role, applied by scripts/20_harden_db.py
-- to every database the agent can connect to, and to template1 so that databases created
-- later start hardened. Idempotent: REVOKE of a privilege already absent changes nothing.
--
-- The role's privileges stop writes to tables, but PostgreSQL grants EXECUTE on many system
-- functions, and SELECT on many system views, to PUBLIC. The ones below would let the agent's
-- SQL do something other than read the benchmark data, even with no privilege on any table:
--   lo_*, loread, lowrite    large objects: create, write and delete rows of pg_largeobject
--   pg_sleep*                hold a connection for as long as the statement timeout allows
--   pg_advisory*, pg_try_advisory*   take locks that outlive the transaction
--   set_config               change session settings, e.g. the time zone results depend on
--   current_setting          read server settings
--   *_to_xml*                run SQL passed as a string, out of sight of any query checker
--   pg_notify                send messages to other sessions
--   pg_terminate_backend, pg_cancel_backend   end or cancel the role's other sessions
--   version, inet_server_*, inet_client_*, pg_postmaster_start_time, pg_conf_load_time
--                            describe the server
--   pg_stat_get_activity, pg_show_all_settings   the functions behind the views below
-- and the views: sessions and their queries, settings, roles and their settings.
-- The functions a superuser alone may run (pg_read_file, pg_ls_dir, lo_import, ...) are
-- already closed to PUBLIC and need nothing here.

DO $$
DECLARE
    f regprocedure;
BEGIN
    FOR f IN
        SELECT p.oid::regprocedure
        FROM pg_proc p
        WHERE p.pronamespace = 'pg_catalog'::regnamespace
          AND (
              p.proname LIKE ANY (ARRAY[
                  'lo\_%', 'pg\_sleep%', 'pg\_advisory%', 'pg\_try\_advisory%', '%\_to\_xml%',
                  'inet\_server\_%', 'inet\_client\_%'
              ])
              OR p.proname = ANY (ARRAY[
                  'loread', 'lowrite', 'set_config', 'current_setting', 'pg_notify',
                  'pg_terminate_backend', 'pg_cancel_backend', 'version',
                  'pg_postmaster_start_time', 'pg_conf_load_time', 'pg_stat_get_activity',
                  'pg_show_all_settings'
              ])
          )
    LOOP
        EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC', f);
    END LOOP;
END
$$;

REVOKE SELECT ON
    pg_catalog.pg_settings,
    pg_catalog.pg_stat_activity,
    pg_catalog.pg_stat_ssl,
    pg_catalog.pg_stat_gssapi,
    pg_catalog.pg_roles,
    pg_catalog.pg_user,
    pg_catalog.pg_group,
    pg_catalog.pg_auth_members,
    pg_catalog.pg_db_role_setting
FROM PUBLIC;

-- Nothing in the public schema for PUBLIC (PostgreSQL grants USAGE by default): a schema role
-- must see its own schema only. The benchmark loader grants the schemas it fills explicitly.
REVOKE ALL ON SCHEMA public FROM PUBLIC;
