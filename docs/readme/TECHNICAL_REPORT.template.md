# An AI Data Analyst with Auditable Answers: Technical Report

*When an AI analyst answers a question about a bank's database, can it tell you how much to trust the answer, and does its confidence track whether it is right?*

The short, plain-language version is the [README](README.md).

**Status: under construction. Nothing has been measured yet; every result is TBD.**

## Abstract

Text-to-SQL systems are usually judged by accuracy alone: how often the query they write returns the right rows. This project asks whether an analyst agent can tell which of its answers are right. Every answer carries its evidence (the exact SQL, the rows it used and the checks it ran) and a confidence, and the agent declines when unsure. The confidence is evaluated against correctness with risk–coverage curves and with calibration measured on held-out questions, on the BIRD mini-dev benchmark and on a hand-written set of banking questions that no model can have seen in training. A later part adds a statistical guardrail against conclusions the data does not support. The central finding is that… TBD.

## Data

The demo database is the Czech bank data of the PKDD'99 discovery challenge (Berka, 1999), included in BIRD as its `financial` database: real, anonymised accounts, clients, transactions, loans and cards from 1993 to 1998, with Czech codes and encoded fields. The benchmark is BIRD mini-dev (Li et al., 2023), loaded from its PostgreSQL release. A hand-written banking test set, fixed before any agent sees it, covers standard, multi-step, ambiguous, unanswerable, false-premise and causal questions. Row counts and file hashes: TBD.

## Method

TBD.

## Results

TBD.

## Limitations

The project is under construction and no part of the evaluation has run. BIRD's questions are public and may be in the training data of the models evaluated; the hand-written banking set is there to measure that risk, and results on the two are reported separately.

## Dependencies

| Component | Version | Role |
|---|---|---|
| Python | `3.12` | the code, with dependencies locked by uv |
| PostgreSQL | `16` | the databases; the agent connects as a read-only role |
| anthropic | see `uv.lock` | model calls, with an on-disk response cache and a spend ledger |
| ollama | see `uv.lock` | the local-model arm |
| psycopg | see `uv.lock` | database access |
| MLflow, DVC | see `uv.lock` | experiment tracking; data versioning |

<sub>Source: `pyproject.toml`, `uv.lock`, `docker-compose.yml`</sub>

## References

Berka, P. (1999). *Guide to the Financial Data Set*. PKDD'99 Discovery Challenge.

Li, J., et al. (2023). *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs* (BIRD). NeurIPS Datasets and Benchmarks. Data licensed CC BY-SA 4.0.
