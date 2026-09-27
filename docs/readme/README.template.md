# An AI Data Analyst with Auditable Answers

*When an AI analyst answers a question about a bank's database, can it tell you how much to trust the answer, and does its confidence track whether it is right?*

This project builds an AI analyst that answers questions about a bank's database by writing and running SQL. Every answer comes with its evidence and a confidence, and the analyst declines when it is unsure. The project measures whether that confidence can be trusted.

**Status: under construction. The data, its dictionary and the analyst's guarded database tools are in place; the analyst itself has not been evaluated yet, so its results below are TBD.**

**Auditable** means that every answer can be traced to the exact SQL, the rows it used and the checks it ran, and that it comes with a calibrated confidence. It does not mean guaranteed correct.

The full technical report is in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). A replay demo of recorded runs will follow.

## The Idea in Plain Terms

- **Text-to-SQL** turns a question in plain English into a database query. The analyst writes the query, runs it, and answers from the rows it gets back.
- **Execution accuracy** is the share of questions for which the analyst's query returns the same rows as a query written by an expert.
- **Calibration** means the confidence matches reality: of the answers given a confidence of eight in ten, about eight in ten should be right.
- **Declining** means the analyst says it cannot answer reliably, and why, instead of guessing.
- Accuracy tells you how often the analyst is right. This project measures whether it knows *which* of its answers are right, and whether its evidence helps a person catch the ones that are not.

## Key Findings

TBD.

## How It Works

The design, being built in stages:

- **The client database.** Real, anonymised data from a Czech bank (1993–1998), with {{financial_trans_rows}} transactions. It is the standard public relational banking dataset and part of the BIRD benchmark. Its codes are in Czech; translating them into business terms is what an analyst does with any bank's internal codes. A hand-written data dictionary gives every column an English name, a meaning and a unit, and translates all {{financial_code_values}} code values found in the data.
- **The benchmark.** BIRD mini-dev, a public text-to-SQL benchmark of {{bird_questions}} questions over {{bird_databases}} databases, with an expert-written query for every question, so the analyst can be compared with published results. All {{gold_executed}} expert queries run on this project's database.
- **A hand-written banking test set.** Questions written for this project and fixed before the analyst sees them, including ambiguous, unanswerable and false-premise questions. Unlike a public benchmark, they cannot be in any model's training data.
- **The analyst.** An agent loop built directly on the Anthropic SDK, with tools to list and describe tables, look at sample rows, run queries and check chart designs.
- **Two independent guards on the database.** A SQL checker accepts only a single read-only query over the analyst's own tables. Separately, the database itself runs every query under a role that can read that one database's tables and nothing else, and stops it at a time limit. Each guard is tested on its own against the same attacks (see Results).
- **The evaluation.** How accuracy rises as the analyst declines its least confident answers (a risk–coverage curve), and whether its confidence is calibrated on questions it was not tuned on.

## Results

**The analyst's accuracy and confidence:** TBD.

**The database guards.** A suite of {{security_attacks}} attacks tried to change data, run a hidden second statement, disguise SQL with comments or look-alike characters, read settings, files and other databases' tables, change session settings, tie up the server, and do what a planted instruction in the data asks. Each guard ran the suite with the other switched off:

{{table:security_suite}}

- No attack got past either guard on its own ({{security_breaches}} breaches), apart from listing table names, below.
- For runaway queries, "stopped" means cut off by the time and row limits, which stay on whichever guard is switched off. No checker can tell an expensive query from a legitimate one, so these queries are contained, not refused.
- The database cannot hide the names of its tables from a user who can connect. Only the checker stops the {{security_names_listed}} queries that list them; the data behind the names stays out of reach.
- All {{guard_gold_accepted}} expert queries of the benchmark pass the checker, and return exactly the same rows through the analyst's tools.

## Try It

TBD.

## Reproducing the Results

- Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), and Docker.
- The database runs in Docker. The data pipeline downloads the benchmark (about 800 MB), loads it and runs every expert query. Every model response is cached, so replaying the results costs nothing.
- The technical report has the details.

```
uv sync
docker compose up -d --wait
uv run dvc repro
uv run pytest
```

## Limitations

- Work in progress: the analyst has not been evaluated yet.
- The security results are from one run of a fixed set of attacks. They show that each guard stops these attacks, not that no other attack exists.
- BIRD's questions are public and may be in the models' training data. The hand-written banking set exists to check for that.

## Dependencies

| Component | Used for |
|---|---|
| Python, uv | the code and its locked environment |
| PostgreSQL, in Docker | the databases, with a read-only role for the analyst |
| Anthropic Python SDK | the analyst's model calls |
| Ollama | a free local model for comparison |
| psycopg | database access from Python |
| sqlglot | the SQL checker's parser |
| jsonschema, Vega-Lite schema | checking the analyst's chart designs |
| MLflow, DVC | experiment tracking and data versioning |

<sub>Source: `pyproject.toml`, `docker-compose.yml`</sub>

## References

Li, J., et al. (2023). *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs* (BIRD). NeurIPS Datasets and Benchmarks. Data licensed CC BY-SA 4.0.

Berka, P. (1999). *Guide to the Financial Data Set*. PKDD'99 Discovery Challenge.
