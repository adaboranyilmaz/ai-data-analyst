# An AI Data Analyst with Auditable Answers

*When an AI analyst answers a question about a bank's database, can it tell you how much to trust the answer, and does its confidence track whether it is right?*

This project builds an AI analyst that answers questions about a bank's database by writing and running SQL. Every answer comes with its evidence and a confidence, and the analyst declines when it is unsure. The project measures whether that confidence can be trusted.

**Status: under construction. The analyst is built, five designs of it have been compared, and the chosen one has been evaluated on the benchmark and on the hand-written banking set. Calibrating its confidence, setting the point at which it declines, the statistical guardrail and the demo come next.**

**Auditable** means that every answer can be traced to the exact SQL, the rows it used and the checks it ran, and that it comes with a calibrated confidence. It does not mean guaranteed correct.

The full technical report is in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). A replay demo of recorded runs will follow.

## The Idea in Plain Terms

- **Text-to-SQL** turns a question in plain English into a database query. The analyst writes the query, runs it, and answers from the rows it gets back.
- **Execution accuracy** is the share of questions for which the analyst's query returns the same rows as a query written by an expert.
- **Calibration** means the confidence matches reality: of the answers given a confidence of eight in ten, about eight in ten should be right.
- **Declining** means the analyst says it cannot answer reliably, and why, instead of guessing.
- Accuracy tells you how often the analyst is right. This project measures whether it knows *which* of its answers are right, and whether its evidence helps a person catch the ones that are not.

## Key Findings

- **The simplest design won.** Of five designs, from a single call with the whole schema to an agent that explores the database, runs its own queries, corrects itself and votes over three attempts, the single call was the most accurate ({{abl_sonnet_d1_ex}}, against {{abl_sonnet_d3_ex}} for the self-correcting agent and {{abl_sonnet_d4_ex}} with voting) and the cheapest ({{abl_sonnet_d1_cost_q}} a question). It won by the rule fixed before any design ran.
- **Why: exploring made a strong model add things.** With the tools to run its own queries, Claude Sonnet 5 more often returned columns it had looked at but was not asked for, and the benchmark scores an extra column as wrong. Its single-call queries rarely failed, so self-correction had little to fix. The smaller Claude Haiku 4.5, whose queries failed more often, gained from the same tools ({{abl_haiku_d1_ex}} to {{abl_haiku_d3_ex}}).
- **On questions nothing was tuned on**, the chosen analyst answered {{held_ex}} of the benchmark's {{held_questions}} held-out questions correctly {{held_ex_ci}}, at {{held_cost_correct}} per correct answer. BIRD's hints matter: without them, accuracy was {{evidence_gain_pts}} points lower.
- **Its confidence ranks its answers, but it is overconfident.** A right answer usually gets a higher confidence than a wrong one (AUROC {{held_auroc}}), but the stated confidences run higher than the accuracy (calibration error {{held_ece}}). Calibrating them is the next step.
- **On the hand-written banking questions**, which no model can have seen, it answered {{own_ab_ex}} of the standard and multi-step questions correctly, declined {{own_d_success}} of the unanswerable ones and corrected {{own_e_success}} of the false premises, while asking a clarifying question on only {{own_clarify_ab}} of the clear questions.

## How It Works

The design, being built in stages:

- **The client database.** Real, anonymised data from a Czech bank (1993–1998), with {{financial_trans_rows}} transactions. It is the standard public relational banking dataset and part of the BIRD benchmark. Its codes are in Czech; translating them into business terms is what an analyst does with any bank's internal codes. A hand-written data dictionary gives every column an English name, a meaning and a unit, and translates all {{financial_code_values}} code values found in the data.
- **The benchmark.** BIRD mini-dev, a public text-to-SQL benchmark of {{bird_questions}} questions over {{bird_databases}} databases, with an expert-written query for every question, so the analyst can be compared with published results. All {{gold_executed}} expert queries run on this project's database.
- **A hand-written banking test set.** {{own_set_questions}} questions written for this project and fixed before the analyst sees them, including ambiguous, unanswerable and false-premise questions. Unlike a public benchmark, they cannot be in any model's training data.
- **The analyst.** An agent loop built directly on the Anthropic SDK, with tools to list and describe tables, look at sample rows, run queries and check chart designs.
- **Two independent guards on the database.** A SQL checker accepts only a single read-only query over the analyst's own tables. Separately, the database itself runs every query under a role that can read that one database's tables and nothing else, and stops it at a time limit. Each guard is tested on its own against the same attacks (see Results).
- **The evaluation.** Accuracy is scored exactly as BIRD's official evaluator scores it, checked against the official code query by query (see Results). Beyond accuracy: how accuracy rises as the analyst declines its least confident answers (a risk–coverage curve), and whether its confidence is calibrated on questions it was not tuned on. The benchmark's questions are split once, before any run: {{split_pilot}} to write the prompts on, {{split_ablation}} to choose the design and calibrate confidence on, and {{split_held_out}} held out for the reported results. How the design will be chosen, and what is expected, is written down before the first run.

## Results

**Choosing the design.** Five designs, each adding one step to the one before, were run on the {{split_ablation}} questions set aside for choosing, with Claude Sonnet 5 on all five, and Claude Haiku 4.5 and a small local model on some. The winner is the design whose confidence best ranks its answers, measured by the area under the risk–coverage curve (AURC); a cheaper design wins if it is not shown to be worse.

{{table:design_comparison}}

- The single call wins outright: the most accurate and the best-ranked, and the cheapest. Each agentic design was less accurate than it on the same questions: the self-correcting agent by {{abl_d1_minus_d3_pts}} points (the single call's lead, with its interval: {{abl_d1_minus_d3}} {{abl_d1_minus_d3_ci}}).
- Voting over three attempts lowered accuracy ({{abl_d4_minus_d3}} {{abl_d4_minus_d3_ci}} against the single attempt): the attempts share the model's habits, so two can agree on the same mistake and outvote a right answer.
- The small local model reached {{abl_local_d1_ex}} with a single call and {{abl_local_d3_ex}} with the tools, far below the hosted models.

**The chosen analyst on the benchmark.** On the {{held_questions}} held-out questions it answered {{held_ex}} correctly {{held_ex_ci}}: {{held_simple_ex}} of the simple ones and {{held_challenging_ex}} of the challenging ones. Answering only its most confident four in five raises that to {{held_acc_at_80}}. On all {{bird_questions}} questions it scored {{all_ex}}. Without BIRD's hints, which spell out the definitions a question relies on, it scored {{noev_ex}} on the choosing set, {{evidence_gain_pts}} points lower (the hints' gain, with its interval: {{evidence_gain}} {{evidence_gain_ci}}).

**The chosen analyst on the banking set.** It answered {{own_ab_ex}} of the standard and multi-step questions correctly {{own_ab_ex_ci}}, declined {{own_d_success}} of the unanswerable ones with a reason, and corrected {{own_e_success}} of the false premises. A check by hand of what it said confirmed {{own_e_reviewed}} of the corrections; of the ambiguous questions, {{own_c_reviewed}} were handled well after the check ({{own_c_success}} by the automatic rule, which counts any clarifying question). On the clear questions it asked for clarification on {{own_clarify_ab}}.

<sub>Source: `results/metrics/benchmark_main.json`, `results/metrics/own_set.json`, `results/reviews/own_set_review.yaml` (one run each)</sub>

**Checking the scorer.** Before scoring the analyst, the project's scoring was run beside BIRD's official evaluator on {{ex_validation_cases}} test queries: the {{ex_validation_gold}} expert queries themselves, {{ex_validation_mutants}} expert queries altered on purpose (a missing DISTINCT, a flipped sort, a dropped filter, a cast to another type) and {{ex_validation_edge}} hand-written edge cases.

- The two gave the same verdict on {{ex_validation_identical}} of the {{ex_validation_cases}}, including the {{ex_validation_mutants_right}} altered queries that still return the right rows. The secondary score (Soft-F1) matched to the last digit on {{soft_f1_validation_identical}}.
- Three kinds of query that the official evaluator runs are refused here by design: a second statement, a setting changed before the query, and another database's tables. The analyst's own tools refuse them before they run, so no answer of the analyst can meet them.
- With PostgreSQL's default parallel query switched on, the official evaluator scored {{parallel_gold_wrong}} of the {{parallel_gold_cases}} expert queries wrong against themselves: their floating-point sums came out differently on each run. Every query here runs without it.

<sub>Source: `results/metrics/ex_validation.json` (one run)</sub>

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

- Every result is from one run; the intervals show how much it could move with other questions, not with another run.
- The benchmark scores an answer's rows exactly: an extra column, or a number of another type, makes an otherwise useful answer wrong. Part of the agentic designs' loss is of this kind.
- The analyst's confidence is not yet calibrated, and it does not yet decline on its own; both come next.
- {{gold_reads_the_clock}} of the benchmark's expert queries compute ages from today's date, so their correct answers change over time. The scorer runs them beside each answer, but a stored answer that fixed a year can go out of date.
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
| NumPy | bootstrap intervals |
| MLflow, DVC | experiment tracking and data versioning |

<sub>Source: `pyproject.toml`, `docker-compose.yml`</sub>

## References

Li, J., et al. (2023). *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs* (BIRD). NeurIPS Datasets and Benchmarks. Data licensed CC BY-SA 4.0.

Berka, P. (1999). *Guide to the Financial Data Set*. PKDD'99 Discovery Challenge.
