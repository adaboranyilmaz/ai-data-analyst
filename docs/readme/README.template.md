# An AI Data Analyst with Auditable Answers

*When an AI analyst answers a question about a bank's database, can it tell you how much to trust the answer?*

An AI analyst answers questions about a bank's database by writing and running SQL. This project measures whether its confidence can be trusted: whether it knows which of its answers are right, where it goes wrong, and what its mistakes cost.

**Auditable** means every answer can be traced to the exact SQL, the rows it used and the checks it ran, and comes with a calibrated confidence. It does not mean guaranteed correct.

- **Live demo:** [adaboranyilmaz.github.io/ai-data-analyst](https://adaboranyilmaz.github.io/ai-data-analyst/) replays recorded runs, with no key and no server behind it.
- **Full technical report:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md), with every table, interval and design decision.

## The Idea in Plain Terms

- **Text-to-SQL** turns a question in plain English into a database query. The analyst writes the query, runs it, and answers from the rows it gets back.
- **Execution accuracy** is the share of questions for which the analyst's query returns the same rows as a query written by an expert.
- **Calibration** means the confidence matches reality: of the answers given a confidence of eight in ten, about eight in ten should be right.
- **Declining** means the analyst says it cannot answer reliably, and why, instead of guessing.
- Accuracy tells you how often the analyst is right. This project measures whether it knows *which* of its answers are right, where and why it goes wrong, and what its mistakes cost.

## Key Findings

1. **The simplest design won.**
   - Of five designs, a single model call with the whole schema was the most accurate ({{abl_sonnet_d1_ex}}) and the cheapest ({{abl_sonnet_d1_cost_q}} a question).
   - An agent that explores the database and corrects itself scored {{abl_sonnet_d3_ex}}; voting over three attempts scored {{abl_sonnet_d4_ex}}.
   - Why: given tools, Claude Sonnet 5 returned columns it had looked at but was not asked for, and the benchmark counts an extra column as wrong. The weaker Claude Haiku 4.5 did gain from the tools ({{abl_haiku_d1_ex}} to {{abl_haiku_d3_ex}}).

2. **Its confidence ranks its answers, and calibration makes it honest.**
   - On {{held_questions}} held-out questions, it answered {{held_ex}} correctly {{held_ex_ci}}, and a right answer usually got a higher confidence than a wrong one (AUROC {{held_auroc}}).
   - Its stated confidence ran too high (calibration error {{cal_raw_ece}}). Adjusted on other questions, it matched reality closely ({{cal_platt_ece}}).
   - But declining to reach {{decline_target}} accuracy left only {{decline_held_coverage}} of the questions answered.

3. **A larger model was far more accurate, and became the served analyst through a rule written in advance.**
   - Claude Opus 5.5 answered {{router_opus_ex}} of the held-out questions correctly, against {{router_sonnet_ex}} for Claude Sonnet 5.
   - The served system, a router that sends Opus the questions Sonnet is unsure of, passed the promotion rule: {{promo_challenger_ex}} against {{promo_champion_ex}} ({{promo_ex_diff}}, {{promo_ex_diff_ci}}).
   - The trade-off: its confidence is less well calibrated ({{promo_challenger_ece}} against {{promo_champion_ece}}), and a question costs {{promo_challenger_cost}} instead of {{promo_champion_cost}}. Sonnet kept its own answer on only {{router_kept}} of {{router_questions}} questions, so Opus alone was as accurate and cheaper.

4. **Most wrong answers are wrong in their logic, and the price of a mistake decides the model.**
   - Of {{ea_wrong}} wrong held-out answers, only {{ea_format_only}} had the right rows in the wrong shape. The rest mostly read other tables ({{ea_tables}}), filtered differently ({{ea_filter}}) or calculated differently ({{ea_computation}}).
   - A close reading of {{hc_items}} of them found {{hc_questionable}} where the benchmark's own expert query does not answer the question as asked.
   - Once a wrong answer costs more than {{da_break_even}}, the larger model is the cheaper system, despite its price ({{da_opus_api}} against {{da_sonnet_api}} a question). The router is never the cheapest.

5. **On hand-written banking questions that no model can have seen, it held up.**
   - It answered {{own_ab_ex}} of the standard and multi-step questions correctly.
   - It declined {{own_d_success}} of the unanswerable ones and corrected {{own_e_success}} of the false premises.
   - It asked for clarification on only {{own_clarify_ab}} of the clear questions.

6. **A statistical guardrail keeps answers true to the test, but not the test true to the question.**
   - On copies of the bank's data with effects planted in them, answers that saw the test claimed an effect where none was planted on {{l2_none_guarded}} of the copies, against {{l2_none_numbers}} for answers written from the numbers alone.
   - But the analyst never chose to compare within a planted third factor, so its test reported a false effect on {{pl_ana_confounded}} of those copies, and the answers repeated it.
   - On the banking set's {{gr_f_questions}} comparative and causal questions, {{gb_ran}} answers came with an interval and the caveat that an association is not a cause. Without the guardrail, none had an interval.

![Risk-coverage curves on the held-out questions](results/plots/risk_coverage_held_out.png)
*How often the analyst is wrong among the answers it keeps, keeping its most confident answers first (lower is better). Its own calibrated confidence (blue) and a second model's review (orange) rank the answers about equally well; the two combined (green, exploratory) do a little better. The dotted line is a perfect ranking. One run, {{held_questions}} held-out questions.*

## How It Works

### The data

- **The client database:** real, anonymized data from a Czech bank (1993–1998), with {{financial_trans_rows}} transactions. It is the standard public relational banking dataset and part of the BIRD benchmark. Its codes are in Czech; translating them into business terms is what an analyst does with any bank's internal codes.
- **A data dictionary,** written by hand, gives every column an English name, a meaning and a unit, and translates all {{financial_code_values}} code values found in the data.
- **The benchmark:** BIRD mini-dev, {{bird_questions}} public text-to-SQL questions over {{bird_databases}} databases, each with an expert-written query.
- **A hand-written banking test set:** {{own_set_questions}} questions, fixed before the analyst saw them, including ambiguous, unanswerable and false-premise ones. They cannot be in any model's training data.

### The analyst

- **An agent loop** built directly on the Anthropic SDK, with tools to list and describe tables, look at sample rows, run queries and check chart designs.
- **Five designs**, each adding one step: a single call with the whole schema; schema tools; self-correction; a vote over three attempts; schema narrowing.
- **Two independent guards on the database.** A SQL checker accepts only a single read-only query. Separately, the database runs every query under a role that can only read, with a time limit. Each guard is tested on its own against the same attacks.
- **A statistical guardrail** for comparative and causal questions: a planned analysis, computed by a tested program in a locked-down container.

### The evaluation

- **Scored as BIRD's official evaluator scores it,** checked against the official code query by query.
- **The questions were split once, before any run:** {{split_pilot}} to write the prompts on, {{split_ablation}} to choose the design and calibrate confidence on, and {{split_held_out}} held out for the reported results.
- **Beyond accuracy:** a risk–coverage curve (how accuracy rises as the analyst declines its least confident answers) and calibration on questions it was not tuned on.
- **Predictions first:** how the design would be chosen, and what was expected, was written down before the first run.

## Results

### Choosing the design

Five designs were run on the {{split_ablation}} questions set aside for choosing. The winner is the design whose confidence best ranks its answers (the area under the risk–coverage curve, AURC); a cheaper design wins if it is not shown to be worse.

{{table:design_comparison}}

- The single call wins outright: the most accurate, the best at ranking, and the cheapest.
- The self-correcting agent was {{abl_d1_minus_d3_pts}} points less accurate on the same questions.
- Voting lowered accuracy: the attempts share the model's habits, so two can agree on the same mistake and outvote a right answer.

### Calibrating and declining

- The stated confidence ran too high: calibration error {{cal_raw_ece}}, and {{cal_platt_ece}} {{cal_platt_ece_ci}} after calibration.
- To be right {{decline_target}} of the time, the analyst may answer only at a stated confidence of {{decline_raw_threshold}} or more. That meant answering {{decline_held_coverage}} of the held-out questions, {{decline_held_accuracy}} of them correctly.
- A second model reviewing the SQL and its rows ranked the answers no better than the analyst's own confidence.
- {{nm_near}} of {{nm_wrong}} wrong answers whose query ran held the right rows with extra or reordered columns. Counted as right, accuracy would rise from {{held_ex}} to {{held_ex_up_to_format}}.

<sub>Source: `results/metrics/calibration.json`, `results/metrics/risk_coverage.json`, `results/metrics/near_miss.json` (one run each)</sub>

### The larger model and the router

- On the questions set aside for choosing, Claude Opus 5.5 was {{esc_gain}} {{esc_gain_ci}} more accurate than Claude Sonnet 5, so the rule fixed in advance adopted it.
- On the held-out questions, the router answered {{router_ex}} correctly against {{router_sonnet_ex}} for Sonnet alone, at {{router_cost_correct}} per correct answer against {{router_sonnet_cost_correct}}.
- Opus alone scored the same, {{router_opus_ex}}, for less: routing pays for a Sonnet answer first and then sets most of them aside.

<sub>Source: `results/metrics/escalation.json`, `results/metrics/router.json` (one run per model)</sub>

### The database guards

A suite of {{security_attacks}} attacks tried to change data, hide a second statement, disguise SQL, read files and other databases' tables, tie up the server, and do what a planted instruction in the data asks. Each guard ran the suite with the other switched off:

{{table:security_suite}}

- No attack got past either guard on its own ({{security_breaches}} breaches), apart from listing table names.
- Runaway queries are cut off by the time and row limits, which stay on whichever guard is switched off.
- All {{guard_gold_accepted}} expert queries of the benchmark pass the checker and return exactly the same rows through the analyst's tools.

### Other checks

- **The scorer:** it gave the same verdict as BIRD's official evaluator on {{ex_validation_identical}} of {{ex_validation_cases}} test queries, including expert queries altered on purpose.
- **A framework version:** rebuilt with LangGraph, the pipeline sent exactly the same requests on all {{fw_identical}} questions, so its answers were identical. The hand-written loop stays the main system.

<sub>Source: `results/metrics/ex_validation.json`, `results/metrics/framework_comparison.json` (one run each)</sub>

## Where the Analyst Gets It Wrong

- **No result:** {{ea_no_result}} of the {{ea_wrong}} wrong held-out answers returned nothing ({{ea_refused}} refused by the SQL checker, {{ea_failed}} failed).
- **Wrong shape:** {{ea_format_only}} had the right rows in the wrong shape.
- **Wrong logic:** {{ea_logic}} read other tables ({{ea_tables}}), used other conditions ({{ea_filter}}), calculated differently ({{ea_computation}}), returned other columns ({{ea_output}}), joined wrongly ({{ea_join}}) or kept other rows ({{ea_order_limit}}).
- **Sometimes the benchmark is wrong:** a close reading of {{hc_items}} found {{hc_questionable}} whose expert query does not answer the question as asked, for example counting lab records where the question asks for patients.

![Where the wrong answers go wrong](results/plots/errors_by_category.png)

<sub>Source: `results/metrics/error_analysis.json`, `results/reviews/error_hand_check.yaml` (one run)</sub>

**What errors cost.** Nobody knows in general what a wrong answer costs, so no price is assumed. Each system's expected cost per question was worked out for every pair of costs, from a cent to $10,000 for a wrong answer and from a cent to $1,000 for a declined question:

- Opus alone is cheaper than Sonnet alone once a wrong answer costs more than {{da_break_even}} (95% interval {{da_break_even_low}} to {{da_break_even_high}}).
- The router is never the cheapest: it gets the same questions wrong as Opus alone and costs {{da_router_extra}} more per question.
- Declining pays only when a person's answer is cheap next to a wrong one: for Sonnet, below {{da_sonnet_pays}} times the cost of a wrong answer.

![The cheapest system at each cost of a wrong answer and of declining](results/plots/cheapest_system.png)
*The system with the lowest expected cost per question, at each cost of a wrong answer (across) and of a declined question (up). Blue is Sonnet and orange is Opus; hatched areas decline their least confident questions. One run per model, {{held_questions}} held-out questions, batch prices.*

<sub>Source: `results/metrics/decision_analysis.json` (one run per model, no model calls)</sub>

## The Statistical Guardrail

Some questions ask whether something holds, not what the rows say: do loans to women go bad more often? Answered from the rows alone, such a question invites a confident claim the data cannot support. These questions take a second path:

1. **Flag:** keyword rules and Claude Haiku 4.5 flag the question as statistical.
2. **Plan:** Claude Sonnet 5, reading the schema but no data, plans the analysis: one row per unit, the outcome, the groups, and up to two variables to compare within.
3. **Analyze:** a tested statistics program computes the intervals and tests in a container with no network, a read-only file system and no privileges.
4. **Answer:** Sonnet writes the answer from that result, with the interval, any warning, and the caveat that records nobody assigned at random show associations, not causes.

What it showed, on copies of the bank's loans with effects of known size planted in them:

- **The test is sound.** With plans written beforehand, it found effects where none were planted in {{pl_ref_false_alarms}} of the copies (it is built for one in twenty) and saw through a planted third factor.
- **The analyst's plans are the weak point.** It never chose to compare within the planted third factor, so its test reported the false effect on {{pl_ana_confounded}} of those copies.
- **The answers follow the test.** Guarded answers never contradicted it ({{l2_disagree_guarded}} of {{l2_disagree_guarded_n}}); answers from the numbers alone did {{l2_disagree_numbers}} times.
- **The sandbox held.** {{sb_blocked}} of {{sb_attacks}} attacks on it were blocked, and each protection, switched off, let its attack through.

![What the answers claim](results/plots/planted_claims.png)
*The share of answers that claim an effect where none was planted (top two rows) and that claim the planted direction where one was (bottom three), from the numbers alone (gray) and guarded (blue). One run, {{pl_copies2}} copies per question and condition.*

<sub>Source: `results/metrics/guardrail.json`, `results/metrics/planted_effects.json`, `results/metrics/sandbox_security.json` (one run each)</sub>

## Operating It

The analyst runs as a service, so it is tracked, versioned, tested before release and watched while it runs.

- **Traces in two tools:** every run's model and tool calls go over OpenTelemetry to MLflow and to a self-hosted Langfuse. Langfuse shows tokens and cost per model call; MLflow keeps the traces beside {{mlflow_runs_logged}} logged experiment runs.
- **A registry:** an agent configuration (prompts, model settings, calibrators, decline threshold, router) is hashed as one version. The service loads the version the registry names as champion.
- **A promotion rule,** written before the challenger's ranking was computed, decides which version is served, and every decision is logged.
- **A gate in CI:** {{gate_unchanged}} of {{gate_entries}} recorded model requests are rebuilt from the repository and must match, so a prompt or schema change fails the build. It needs no database and no key.
- **A canary:** a manual, capped workflow resends {{canary_calls}} fixed requests to the model. The first run cost {{canary_spent}}; {{canary_valid}} of the answers were valid and {{canary_sql_same}} had the same query.
- **Drift and alerts:** the service compares the confidence of its last {{drift_window}} answers with the evaluation's. On windows drawn from the held-out questions themselves, it alerted on {{drift_ref_alert}}; replayed shifted traffic fired the Prometheus alert, which then cleared.
- **Load:** the replay service answered every request of a load test with {{load_errors}} errors.

![The same recorded run in Langfuse](results/plots/trace_langfuse.png)
*One recorded run of the self-correcting agent in Langfuse, sent from the stored spans: each model call with its tokens and cost. The same run in MLflow is in `results/plots/trace_mlflow.png`.*

The design is drawn in [docs/architecture.md](docs/architecture.md).

## Try It

- **Live demo:** [adaboranyilmaz.github.io/ai-data-analyst](https://adaboranyilmaz.github.io/ai-data-analyst/). It replays {{static_runs}} recorded runs as they streamed: the steps, the query in words, the rows, the checks, the answer and what its confidence means. The runs were drawn by a rule fixed in advance, so wrong and held-back answers are in the set.
- **Locally, recorded runs (no key):** `docker compose up -d --wait`, then open http://127.0.0.1:8000.
- **Locally, your own questions:** put an API key in `.env`, load the benchmark, and run `uv run python -m src.serving --mode live`. A question costs a few cents, under a spending cap. It is for one person on their own machine: no sign-in, localhost only.
- **Your own PostgreSQL:** in live mode the page can connect to your database after checking that the role can only read. The confidence is not calibrated for your data, and the page says so.

<sub>Source: `results/metrics/serving_check.json`, `results/metrics/static_site.json`</sub>

### Use it from an MCP client

The same guarded, read-only tools work in any [MCP](https://modelcontextprotocol.io) client: list tables, describe one, sample rows, look up a code in the dictionary, and run one `SELECT`. Start the database (`docker compose up -d --wait postgres`) and add this to the client's configuration:

```json
{
  "mcpServers": {
    "ai-data-analyst": {
      "command": "uv",
      "args": ["run", "--directory", "C:/path/to/ai-data-analyst", "python", "-m", "src.mcp_server"]
    }
  }
}
```

- To serve your own PostgreSQL, set `ANALYST_MCP_PG_HOST`, `ANALYST_MCP_PG_PORT`, `ANALYST_MCP_PG_DBNAME`, `ANALYST_MCP_PG_USER`, `ANALYST_MCP_PG_PASSWORD` and, if needed, `ANALYST_MCP_PG_SCHEMA` in the server's `env`. A role that can write is refused.
- The security suite was sent through this server by a real MCP client: {{mcp_attacks}} attacks on the query tool and {{mcp_boundary_attacks}} on the tools' arguments, with {{mcp_breaches}} breaches.

<sub>Source: `results/metrics/mcp_security.json` (one run)</sub>

## Reproducing the Results

- **Install:** Python 3.12 with [uv](https://docs.astral.sh/uv/), and Docker for the database.
- **Data:** the pipeline downloads the benchmark (about 800 MB), loads it and runs every expert query. Every model response is cached, so rebuilding costs nothing.
- **Checks:** a forced rebuild of a clean copy reproduced every results file, apart from timings and a few row-order effects listed in the report.

```bash
uv sync
docker compose up -d --wait postgres          # the database
ANALYST_REPLAY_ONLY=1 uv run dvc repro        # rebuild every result, no AI calls
uv run pytest                                 # the tests
uv run python scripts/90_readme.py --check    # confirm these documents match the results
```

The technical report lists every step.

## Limitations

- **Single runs.** Every result is from one run. The intervals show how much it could move with other questions, not with another run.
- **A strict benchmark.** An extra column or a number of another type makes an otherwise useful answer wrong.
- **Small calibration set.** The calibration and the decline threshold rest on {{split_ablation}} questions.
- **Simple costs.** The cost analysis prices every wrong answer the same and assumes a person answers a declined question correctly.
- **Known attacks only.** The security suites show that each guard stops these attacks, not that no other attack exists.
- **The guardrail depends on the plan.** It computes the planned test correctly but cannot see a third factor the plan leaves out.
- **Possible contamination.** BIRD's questions are public and may be in the models' training data. The hand-written banking set exists to check for that.
- **Not production-ready.** The live service has no sign-in or rate limit beyond its spending cap, and replayed traffic is not live traffic.

## Dependencies

| Package | Purpose |
|---|---|
| anthropic | The analyst's model calls |
| ollama | A free local model for comparison |
| psycopg, sqlglot | Database access and the SQL checker's parser |
| jsonschema | Checking the analyst's chart designs |
| numpy, scipy, statsmodels, pandas | Intervals; the guardrail's statistics, in a locked-down container |
| mlflow, dvc | Experiment tracking, the registry view and the versioned data pipeline |
| opentelemetry | Traces sent to MLflow and Langfuse |
| langgraph | The framework comparison only |
| fastapi, uvicorn, prometheus-client | The service and its metrics |
| mcp | The tool server for MCP clients |
| matplotlib | Every plot |
| React, Vite, Vega-Embed | The page |
| Langfuse, Prometheus, Grafana | Tracing, alerts and the dashboard, in Docker |

<sub>Source: `pyproject.toml`, `ui/package.json`, `docker-compose.yml`</sub>

## References

Berka, P. (1999). *Guide to the Financial Data Set*. PKDD'99 Discovery Challenge.

Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*. NeurIPS.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML.

Li, J., et al. (2023). *Can LLM Already Serve as a Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs* (BIRD). NeurIPS Datasets and Benchmarks. Data licensed CC BY-SA 4.0.

Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. Advances in Large Margin Classifiers, MIT Press.
