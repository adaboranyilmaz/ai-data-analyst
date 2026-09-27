# An AI Data Analyst with Auditable Answers: Technical Report

*When an AI analyst answers a question about a bank's database, can it tell you how much to trust the answer, and does its confidence track whether it is right?*

The short, plain-language version is the [README](README.md).

**Status: under construction. The data layer and the analyst's guarded tools are in place; the analyst has not been evaluated, so its results are TBD.**

## Abstract

Text-to-SQL systems are usually judged by accuracy alone: how often the query they write returns the right rows. This project asks whether an analyst agent can tell which of its answers are right. Every answer carries its evidence (the exact SQL, the rows it used and the checks it ran) and a confidence, and the agent declines when unsure. The confidence is evaluated against correctness with risk–coverage curves and with calibration measured on held-out questions, on the BIRD mini-dev benchmark and on a hand-written set of banking questions that no model can have seen in training. A later part adds a statistical guardrail against conclusions the data does not support. The central finding is that… TBD.

## Data

The demo database is the Czech bank data of the PKDD'99 discovery challenge (Berka, 1999), included in BIRD as its `financial` database: real, anonymised accounts, clients, transactions, loans and cards from 1993 to 1998, with Czech codes and encoded fields. The benchmark is BIRD mini-dev (Li et al., 2023): {{bird_questions}} questions over {{bird_databases}} databases ({{bird_tables}} tables, {{bird_columns}} columns). A hand-written banking test set, fixed before any agent sees it, covers standard, multi-step, ambiguous, unanswerable, false-premise and causal questions.

The databases come from BIRD's PostgreSQL dump, loaded into PostgreSQL 16 with one schema per database; the questions and their expert SQL come from BIRD's Hugging Face release, which BIRD names canonical, pinned to one revision. The older copy of the questions inside the download package lists two questions twice and misses two others, and two of its expert queries differ from the current ones. Every source file is pinned by hash. The Czech bank tables match the row counts published with the data, table by table ({{financial_trans_rows}} transactions).

Every expert query was executed as the analyst's read-only role: {{gold_executed}} of {{bird_questions}} ran, and {{gold_excluded}} were excluded. Each ran twice, and again after the tables moved into per-database schemas; every result set was the same each time. That holds only for serial execution: with PostgreSQL's parallel workers, a floating-point sum adds its terms in a different order on each run, and two expert answers (a sum and an average over a single-precision column) changed from run to run, one of them in its fifth significant digit. The expert queries therefore run without parallel workers, and the analyst's queries will run the same way. The time zone matters too: the dump's timestamps were written at UTC+8, and the benchmark's questions and expert SQL use those local times, so the database runs in that zone; in UTC, five expert answers differ, two of them empty.

The Czech bank has a hand-written data dictionary: English names, meanings, units, join paths, known quirks, and translations of all {{financial_code_values_translated}} Czech code values present in the data, checked by tests against the data. Profiling found what neither the published guide nor BIRD's descriptions say: a third transaction type (`VYBER`, a withdrawal), the bank's own fees and penalty interest booked under the operation the guide calls a cash withdrawal, two spellings of "no purpose" (NULL and a single space), district statistics stored as text, and money labelled in dollars in BIRD's descriptions although it is in Czech koruna. For the other ten databases, BIRD's own column descriptions are converted unchanged into the same format; {{bird_columns_described}} of the {{bird_columns}} columns across all eleven databases have a description.

<sub>Source: `results/metrics/data_stats.json`, `results/metrics/bird_sources.json`, `results/metrics/gold_execution.json`</sub>

## Method

### The analyst's tools and their two guards

The analyst works through five tools: `list_tables` and `describe_table` (the schema with each column's profile and its data-dictionary entry, read from committed files rather than the database's catalogs), `sample_rows` (a few whole rows, chosen by a hash of each row's contents, so a replayed run sees the same sample), `run_sql`, and `validate_chart` (a Vega-Lite chart design checked against the official schema, allowed to show only the query's own result, with nothing a browser would fetch: no URLs, no links). The database is bound by the caller, never chosen by the model.

Every query passes two independent guards, and each must stop every attack on its own. The first is a SQL checker. It works in two passes. A lexical pass follows PostgreSQL's own rules for strings and comments and refuses anything whose reading depends on them: comments, backslashes, dollar quoting, escape strings, a second statement, and characters outside printable ASCII except inside quotes (where invisible and control characters are still refused). A parser that disagrees with PostgreSQL about where a string ends would otherwise let text that looks like data run as SQL. The same pass refuses system objects and a list of server functions by name. A structural pass on a syntax tree (sqlglot) then accepts only a query (`SELECT`, a set operation, or either under `WITH`) over the tables of the analyst's own database, with every join on a stated condition and bounded sizes for the functions that can build very large results.

The second guard is the database. Each query runs through a server-side cursor, which PostgreSQL accepts for a single query only, in a read-only transaction that is always rolled back, under a role that can read one benchmark database's tables and nothing else: the connecting role switches to it inside each transaction, and nothing the query can run switches back. There are {{schema_roles}} such roles, one per benchmark database. The hardening also closes {{hardened_functions}} system functions and {{hardened_views}} system views that PostgreSQL leaves open to every role by default: large-object functions (which write without any table privilege), sleeping, advisory locks, changing settings, running SQL passed as text, and views of settings, sessions and roles. Each query also sets its own search path, time zone and planner settings, so nothing an earlier query did carries over, and the results match the expert SQL's execution exactly. Two limits hold whichever guard is switched off: a timer on the client cancels a query at its time limit (the role could lift a server-side timeout itself), and at most the row limit plus one row leaves the server, with the rest counted there.

The security suite runs {{security_attacks}} attacks three times: against the checker alone (its verdict; a resource attack it accepts runs as the administrator, with no database guard, under the limits), against the database alone (the checker bypassed), and against the role's privileges alone (also without the read-only transaction). After every attack, whatever its outcome, the suite checks what could have changed: a canary table, the relations and large objects in the database, grants, advisory locks, queries left running, a secret token anywhere in the result, and the next query's time zone, role and search path. The prompt-injection attacks are the SQL a planted row asks for; whether a model obeys such text is tested with the agent.

## Results

**The analyst's accuracy and confidence:** TBD.

### The database guards

{{table:security_suite}}

No attack got past either guard on its own ({{security_breaches}} breaches), apart from the listing of object names described below. The runaway queries count as stopped when they were contained: cut off at the suite's {{security_timeout}}-second limit, refused by PostgreSQL's limit on temporary files, or cut at the row limit. The checker also refuses the explicit patterns (cross joins, recursive queries, unbounded series), but a legitimate-looking join can still run long, and only the limits stop it. The one exception is by design and recorded as such: PostgreSQL shows object names to any role that can connect, through its catalogs, casts to `regclass` and its error messages, so the {{security_names_listed}} attacks that list schemas and tables are stopped by the checker only. None of them reaches data.

Building the suite found two gaps in the database guard as first configured, both closed before these results. The large-object functions were open to every role and write without any table privilege, so with the read-only transaction switched off, the role's privileges alone did not stop them. And one role read all eleven benchmark databases, so the database did not confine a query to its own; the per-database roles do.

The checker accepts all {{guard_gold_accepted}} expert queries unchanged, and all {{tools_gold_reproduced}} return exactly the stored expert results through the analyst's execution path. A query through `run_sql` took {{run_sql_p50_ms}} ms at the median and {{run_sql_p95_ms}} ms at the 95th percentile over the {{run_sql_calls}} expert queries; `sample_rows` took {{sample_rows_p50_ms}} ms at the median but up to {{sample_rows_max_ms}} ms on the largest table, whose million rows are hashed to choose the sample; validating a chart took {{validate_chart_p50_ms}} ms (one run, local database).

<sub>Source: `results/metrics/security_suite.json`, `results/metrics/db_hardening.json`, `results/metrics/tool_check.json` (one run)</sub>

## Limitations

The project is under construction and the analyst has not been evaluated. The security suite is one run of a fixed set of attacks: it shows that each guard stops these attacks on its own, not that no other attack exists, and the checker's lexical rules are written for PostgreSQL alone. PostgreSQL has no per-query memory limit, so a query that builds one very large value is bounded by the checker's size limits and by PostgreSQL's one-gigabyte limit on a single value, not by the database guard. BIRD's questions are public and may be in the training data of the models evaluated; the hand-written banking set is there to measure that risk, and results on the two are reported separately.

## Dependencies

| Component | Version | Role |
|---|---|---|
| Python | `3.12` | the code, with dependencies locked by uv |
| PostgreSQL | `16` | the databases; the agent connects as a read-only role |
| anthropic | see `uv.lock` | model calls, with an on-disk response cache and a spend ledger |
| ollama | see `uv.lock` | the local-model arm |
| psycopg | see `uv.lock` | database access |
| sqlglot | see `uv.lock` | the SQL checker's syntax tree |
| jsonschema | see `uv.lock` | chart validation |
| Vega-Lite JSON schema | `6.4.3`, vendored and pinned by hash | the chart grammar |
| MLflow, DVC | see `uv.lock` | experiment tracking; data versioning |

<sub>Source: `pyproject.toml`, `uv.lock`, `docker-compose.yml`</sub>

## References

Berka, P. (1999). *Guide to the Financial Data Set*. PKDD'99 Discovery Challenge.

Greshake, K., et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*. AISec.

Li, J., et al. (2023). *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs* (BIRD). NeurIPS Datasets and Benchmarks. Data licensed CC BY-SA 4.0.

Satyanarayan, A., Moritz, D., Wongsuphasawat, K., & Heer, J. (2017). *Vega-Lite: A Grammar of Interactive Graphics*. IEEE Transactions on Visualization and Computer Graphics.
